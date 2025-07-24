import xml.etree.ElementTree as ET
import json
from typing import List, Dict, Any
import re
import os

class MusicXMLProcessor:
    def __init__(self):
        # Map note durations to more readable tokens
        self.duration_map = {
            6: "16th",
            12: "8th", 
            24: "quarter",
            48: "half",
            96: "whole"
        }
        
    def extract_musical_data(self, xml_content: str) -> List[Dict[str, Any]]:
        """Extract essential musical data from MusicXML content"""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return []
        
        musical_events = []
        current_measure = 1
        
        # Find all measures
        for measure in root.findall(".//measure"):
            measure_num = measure.get('number', str(current_measure))
            
            # Extract time signature and key signature from first measure
            if current_measure == 1:
                time_sig = self._extract_time_signature(measure)
                key_sig = self._extract_key_signature(measure)
                if time_sig:
                    musical_events.append({
                        'type': 'time_signature',
                        'measure': self._safe_int(measure_num),
                        'beats': time_sig['beats'],
                        'beat_type': time_sig['beat_type']
                    })
                if key_sig is not None:
                    musical_events.append({
                        'type': 'key_signature', 
                        'measure': self._safe_int(measure_num),
                        'fifths': key_sig
                    })
            
            # Extract notes
            for note in measure.findall(".//note"):
                note_data = self._extract_note_data(note, measure_num)
                if note_data:
                    musical_events.append(note_data)
            
            current_measure += 1
            
        return musical_events
    
    def _safe_int(self, measure_num: str) -> int:
        """Safely convert measure number to int"""
        try:
            return int(measure_num)
        except ValueError:
            # Handle non-numeric measure numbers (like "X1", "A", etc.)
            digits = re.findall(r'\d+', measure_num)
            if digits:
                return int(digits[0])  # Use first number found
            else:
                return hash(measure_num) % 1000  # Convert to consistent number
    
    def _extract_time_signature(self, measure) -> Dict[str, int]:
        """Extract time signature from measure"""
        time_elem = measure.find(".//time")
        if time_elem is not None:
            beats = time_elem.find("beats")
            beat_type = time_elem.find("beat-type")
            if beats is not None and beat_type is not None:
                return {
                    'beats': int(beats.text),
                    'beat_type': int(beat_type.text)
                }
        return None
    
    def _extract_key_signature(self, measure) -> int:
        """Extract key signature (number of sharps/flats)"""
        key_elem = measure.find(".//key/fifths")
        if key_elem is not None:
            return int(key_elem.text)
        return None
    
    def _extract_note_data(self, note_elem, measure_num: str) -> Dict[str, Any]:
        """Extract essential data from a note element"""
        measure_int = self._safe_int(measure_num)
        
        note_data = {
            'type': 'note',
            'measure': measure_int,
            'measure_id': measure_num  # Keep original for reference
        }
        
        # Check if it's a rest
        rest = note_elem.find("rest")
        if rest is not None:
            note_data['pitch'] = 'REST'
            note_data['octave'] = 0
        else:
            # Extract pitch information
            pitch = note_elem.find("pitch")
            if pitch is not None:
                step = pitch.find("step")
                octave = pitch.find("octave")
                alter = pitch.find("alter")
                
                if step is not None and octave is not None:
                    note_data['pitch'] = step.text
                    note_data['octave'] = int(octave.text)
                    
                    # Handle accidentals (sharps/flats)
                    if alter is not None:
                        alter_val = int(alter.text)
                        if alter_val == 1:
                            note_data['pitch'] += '#'
                        elif alter_val == -1:
                            note_data['pitch'] += 'b'
        
        # Extract duration
        duration = note_elem.find("duration")
        if duration is not None:
            dur_val = int(duration.text)
            note_data['duration'] = self.duration_map.get(dur_val, f"dur_{dur_val}")
            note_data['duration_ticks'] = dur_val
        else:
            # If no duration element, try to infer from note type
            note_type = note_elem.find("type")
            if note_type is not None:
                note_data['duration'] = note_type.text
                note_data['duration_ticks'] = 24  # Default quarter note
            else:
                note_data['duration'] = 'quarter'
                note_data['duration_ticks'] = 24
        
        # Extract note type (visual representation)
        note_type = note_elem.find("type")
        if note_type is not None:
            note_data['note_type'] = note_type.text
        
        # Extract stem direction (can be musically significant)
        stem = note_elem.find("stem")
        if stem is not None:
            note_data['stem'] = stem.text
        
        return note_data
    
    def to_sequence_tokens(self, musical_events: List[Dict[str, Any]]) -> List[str]:
        """Convert musical events to a sequence of tokens for transformer training"""
        tokens = []
        
        for event in musical_events:
            try:
                if event['type'] == 'time_signature':
                    tokens.append(f"TIME_{event['beats']}_{event['beat_type']}")
                elif event['type'] == 'key_signature':
                    tokens.append(f"KEY_{event['fifths']}")
                elif event['type'] == 'note':
                    # Handle missing duration gracefully
                    duration = event.get('duration', 'unknown')
                    if duration == 'unknown':
                        # Try to get duration from duration_ticks or note_type
                        if 'duration_ticks' in event:
                            duration = self.duration_map.get(event['duration_ticks'], f"dur_{event['duration_ticks']}")
                        elif 'note_type' in event:
                            duration = event['note_type']
                        else:
                            duration = 'quarter'  # Default fallback
                    
                    if event.get('pitch') == 'REST':
                        tokens.append(f"REST_{duration}")
                    else:
                        pitch = event.get('pitch', 'C')
                        octave = event.get('octave', 4)
                        pitch_octave = f"{pitch}{octave}"
                        tokens.append(f"NOTE_{pitch_octave}_{duration}")
            except Exception as e:
                print(f"Warning: Skipping event due to error: {e}")
                continue
        
        return tokens
    
    def process_file(self, file_path: str, output_format: str = 'tokens') -> str:
        """Process a MusicXML file and return formatted data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
        except FileNotFoundError:
            return f"Error: File {file_path} not found"
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    xml_content = f.read()
            except:
                return "Error: Could not decode file"
        
        musical_events = self.extract_musical_data(xml_content)
        
        if output_format == 'tokens':
            tokens = self.to_sequence_tokens(musical_events)
            return ' '.join(tokens)
        else:
            return "Error: Invalid output format. Use 'tokens'"

def main():
    """Main function to process all .txt files in a subdirectory"""
    processor = MusicXMLProcessor()
    
    try:
        print("Current working directory:", os.getcwd())
        paste_dir_paths = [
            './Transformer/data_file',   

        ]
        
        paste_dir = None
        for dir_path in paste_dir_paths:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                paste_dir = dir_path
                print(f"✓ Found paste files directory: {paste_dir}")
                break
        
        if not paste_dir:
            print("Could not find paste files directory!")
            print("Please create one of these directories and put your .txt files in it:")
            for path in paste_dir_paths[:5]:
                print(f"  {path}")
            return
        
        # Find all .txt files in the directory
        print(f"\nScanning {paste_dir} for .txt files...")
        try:
            all_files = os.listdir(paste_dir)
            txt_files = [f for f in all_files if f.endswith('.txt')]
            txt_files.sort()  # Sort for consistent processing order
            
            print(f"Found {len(txt_files)} .txt files:")
            for f in txt_files:
                print(f"  {f}")
                
        except Exception as e:
            print(f"Error reading directory: {e}")
            return
        
        if not txt_files:
            print(f"❌ No .txt files found in {paste_dir}")
            return
        
        # Process all files and combine tokens
        all_tokens = []
        successfully_processed = []
        total_files = len(txt_files)
        
        print(f"\n🎵 Processing {total_files} Bach files...")
        
        for i, file_name in enumerate(txt_files, 1):
            file_path = os.path.join(paste_dir, file_name)
            print(f"\n[{i}/{total_files}] Processing {file_name}...")
            
            # Process the file
            result = processor.process_file(file_path, 'tokens')
            
            if result.startswith('Error:'):
                print(f"  ❌ Error processing {file_name}: {result}")
                continue
            
            # Add tokens from this file
            file_tokens = result.split()
            all_tokens.extend(file_tokens)
            successfully_processed.append(file_name)
            
            print(f"  ✅ Successfully processed: {len(file_tokens)} tokens")
        
        if not all_tokens:
            print(f"\n❌ No files were successfully processed!")
            return
        
        # Combine all tokens
        combined_result = ' '.join(all_tokens)
        total_tokens = len(all_tokens)
        
        print(f"\n{'='*50}")
        print(f"🎼 COMBINED PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Files processed: {len(successfully_processed)}/{total_files}")
        print(f"Successfully processed files:")
        for file_name in successfully_processed:
            print(f"  ✓ {file_name}")
        
        if len(successfully_processed) < total_files:
            failed_files = set(txt_files) - set(successfully_processed)
            print(f"Failed files:")
            for file_name in failed_files:
                print(f"  ✗ {file_name}")
        
        print(f"\nTotal tokens generated: {total_tokens:,}")
        
        # Show vocabulary statistics
        unique_tokens = sorted(list(set(all_tokens)))
        vocab_size = len(unique_tokens)
        print(f"Unique token vocabulary: {vocab_size:,}")
        
        # Show token distribution by type
        note_tokens = [t for t in all_tokens if t.startswith('NOTE_')]
        rest_tokens = [t for t in all_tokens if t.startswith('REST_')]
        key_tokens = [t for t in all_tokens if t.startswith('KEY_')]
        time_tokens = [t for t in all_tokens if t.startswith('TIME_')]
        
        print(f"\n📊 Token breakdown:")
        print(f"  🎵 Notes: {len(note_tokens):,} ({len(note_tokens)/total_tokens*100:.1f}%)")
        print(f"  🔇 Rests: {len(rest_tokens):,} ({len(rest_tokens)/total_tokens*100:.1f}%)")
        print(f"  🎼 Key signatures: {len(key_tokens):,}")
        print(f"  ⏱️  Time signatures: {len(time_tokens):,}")
        
        # Show sample of unique tokens
        print(f"\n🎯 Sample vocabulary (first 15 unique tokens):")
        for i, token in enumerate(unique_tokens[:15]):
            print(f"  {i+1:2d}: {token}")
        if len(unique_tokens) > 15:
            print(f"  ... and {len(unique_tokens)-15:,} more unique tokens")
        
        # Show first and last tokens
        if total_tokens <= 20:
            print(f"\n=== ALL {total_tokens} TOKENS ===")
            for i, token in enumerate(all_tokens, 1):
                print(f"{i:3d}: {token}")
        else:
            print(f"\n🎼 First 10 tokens:")
            for i, token in enumerate(all_tokens[:10], 1):
                print(f"  {i:2d}: {token}")
            
            print(f"\n... ({total_tokens - 20:,} tokens in between) ...")
            
            print(f"\n🎼 Last 10 tokens:")
            for i, token in enumerate(all_tokens[-10:], total_tokens - 9):
                print(f"  {i:2d}: {token}")
        
        # Save combined tokens to file
        output_file = './Transformer/music_tokens.txt'
        try:
            with open(output_file, 'w') as f:
                f.write(combined_result)
            print(f"\n✅ All {total_tokens:,} combined tokens saved to '{output_file}'")
        except:
            # Fallback to current directory if Transformer folder doesn't exist
            output_file = 'music_tokens.txt'
            with open(output_file, 'w') as f:
                f.write(combined_result)
            print(f"\n✅ All {total_tokens:,} combined tokens saved to '{output_file}'")
        
        print(f"\n🎼 Ready for transformer training with {len(successfully_processed)} Bach pieces!")
        print("Your AI will now learn from multiple Bach compositions for richer musical patterns!")
        print(f"📈 Training data size: {total_tokens:,} tokens from {vocab_size:,} unique musical elements")
        
    except Exception as e:
        # If ANY error occurs, try to save what we have and exit gracefully
        print(f"\n⚠️  ERROR OCCURRED: {e}")
        print("Attempting to save partial results...")
        
        try:
            # Try to save whatever tokens we managed to process
            if 'all_tokens' in locals() and all_tokens:
                combined_result = ' '.join(all_tokens)
                output_file = './Transformer/music_tokens_partial.txt'
                try:
                    with open(output_file, 'w') as f:
                        f.write(combined_result)
                except:
                    output_file = 'music_tokens_partial.txt'
                    with open(output_file, 'w') as f:
                        f.write(combined_result)
                
                print(f"✅ Partial results saved to '{output_file}'")
                print(f"📊 Saved {len(all_tokens):,} tokens before error occurred")
                if 'successfully_processed' in locals():
                    print(f"📁 From {len(successfully_processed)} successfully processed files")
            else:
                print("❌ No valid tokens to save")
        except Exception as save_error:
            print(f"❌ Could not save partial results: {save_error}")

# This is the ONLY code that runs - no leftover code after this point
if __name__ == "__main__":
    main()
    print("🏁 Script execution finished.")
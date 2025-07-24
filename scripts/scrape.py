import xml.etree.ElementTree as ET
import json
from typing import List, Dict, Any
import re
import os

class MusicXMLProcessor:
    def __init__(self):
        self.duration_map = {
            6: "16th",
            12: "8th", 
            24: "quarter",
            48: "half",
            96: "whole"
        }
        
    def extract_musical_data(self, xml_content: str) -> List[Dict[str, Any]]:
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return []
        
        musical_events = []
        current_measure = 1
        
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
        try:
            return int(measure_num)
        except ValueError:
            digits = re.findall(r'\d+', measure_num)
            if digits:
                return int(digits[0])
            else:
                return hash(measure_num) % 1000
    
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
        key_elem = measure.find(".//key/fifths")
        if key_elem is not None:
            return int(key_elem.text)
        return None
    
    def _extract_note_data(self, note_elem, measure_num: str) -> Dict[str, Any]:
        measure_int = self._safe_int(measure_num)
        
        note_data = {
            'type': 'note',
            'measure': measure_int,
            'measure_id': measure_num
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
            note_type = note_elem.find("type")
            if note_type is not None:
                note_data['duration'] = note_type.text
                note_data['duration_ticks'] = 24  # Default quarter note
            else:
                note_data['duration'] = 'quarter'
                note_data['duration_ticks'] = 24
        
        note_type = note_elem.find("type")
        if note_type is not None:
            note_data['note_type'] = note_type.text
        
        stem = note_elem.find("stem")
        if stem is not None:
            note_data['stem'] = stem.text
        
        return note_data
    
    def to_sequence_tokens(self, musical_events: List[Dict[str, Any]]) -> List[str]:
        tokens = []
        
        for event in musical_events:
            try:
                if event['type'] == 'time_signature':
                    tokens.append(f"TIME_{event['beats']}_{event['beat_type']}")
                elif event['type'] == 'key_signature':
                    tokens.append(f"KEY_{event['fifths']}")
                elif event['type'] == 'note':
                    # Handle missing duration
                    duration = event.get('duration', 'unknown')
                    if duration == 'unknown':
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
                print(f"Skipping event due to error: {e}")
                continue
        
        return tokens
    
    def process_file(self, file_path: str, output_format: str = 'tokens') -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
        except FileNotFoundError:
            return f"Error: File {file_path} not found"
        except UnicodeDecodeError:
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
    processor = MusicXMLProcessor()
    try:
        paste_dir = '../data_file'
        try:
            all_files = os.listdir(paste_dir)
            txt_files = [f for f in all_files if f.endswith('.txt')]
            txt_files.sort()
            
            print(f"Found {len(txt_files)} .txt files:")
                
        except Exception as e:
            print(f"Error reading directory: {e}")
            return
        
        if not txt_files:
            print(f"No .txt files found")
            return
        
        all_tokens = []
        successfully_processed = []
        total_files = len(txt_files)
        
        print(f"processing {total_files} files...")
        
        for i, file_name in enumerate(txt_files, 1):
            file_path = os.path.join(paste_dir, file_name)
            
            result = processor.process_file(file_path, 'tokens')
            
            if result.startswith('Error:'):
                print(f"Error processing {file_name}: {result}")
                continue
            
            file_tokens = result.split()
            all_tokens.extend(file_tokens)
            successfully_processed.append(file_name)
            
            print(f"Successfully processed all tokens")
        
        if not all_tokens:
            print(f"Failed to proccess all tokens")
            return
        
        combined_result = ' '.join(all_tokens)
        total_tokens = len(all_tokens)
        output_file = '../tokens/music_tokens.txt'
        try:
            with open(output_file, 'w') as f:
                f.write(combined_result)
            print(f"\nAll {total_tokens:,} combined tokens saved to '{output_file}'")
        except:
            output_file = 'music_tokens.txt'
            with open(output_file, 'w') as f:
                f.write(combined_result)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
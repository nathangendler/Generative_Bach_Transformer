import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Any
import re
import os

class TokensToMusicXMLConverter:
    def __init__(self):
        # Map durations back to MusicXML values
        self.duration_map = {
            "16th": 6,
            "8th": 12,
            "quarter": 24,
            "half": 48,
            "whole": 96,
            "dur_6": 6,
            "dur_12": 12,
            "dur_24": 24,
            "dur_48": 48,
            "dur_96": 96
        }
        
        # Default musical settings
        self.divisions = 24  # Standard divisions per quarter note
        self.default_key = 0  # C major (no sharps/flats)
        self.default_time_beats = 4
        self.default_time_beat_type = 4
        
    def parse_tokens(self, tokens_text: str) -> List[Dict[str, Any]]:
        """Parse token string into structured musical events"""
        tokens = tokens_text.split()
        events = []
        current_key = self.default_key
        current_time_beats = self.default_time_beats
        current_time_beat_type = self.default_time_beat_type
        
        for token in tokens:
            if token.startswith('KEY_'):
                # Extract key signature
                try:
                    key_value = int(token.split('_')[1])
                    current_key = key_value
                    events.append({
                        'type': 'key_signature',
                        'fifths': key_value
                    })
                except (ValueError, IndexError):
                    continue
                    
            elif token.startswith('TIME_'):
                # Extract time signature
                try:
                    parts = token.split('_')
                    beats = int(parts[1])
                    beat_type = int(parts[2])
                    current_time_beats = beats
                    current_time_beat_type = beat_type
                    events.append({
                        'type': 'time_signature',
                        'beats': beats,
                        'beat_type': beat_type
                    })
                except (ValueError, IndexError):
                    continue
                    
            elif token.startswith('NOTE_'):
                # Parse note: NOTE_E5_16th
                try:
                    parts = token.split('_')
                    pitch_octave = parts[1]  # E5
                    duration = parts[2]      # 16th
                    
                    # Extract pitch and octave
                    pitch_match = re.match(r'([A-G][#b]?)(\d+)', pitch_octave)
                    if pitch_match:
                        pitch = pitch_match.group(1)
                        octave = int(pitch_match.group(2))
                        
                        # Handle accidentals
                        step = pitch[0]
                        alter = 0
                        if len(pitch) > 1:
                            if pitch[1] == '#':
                                alter = 1
                            elif pitch[1] == 'b':
                                alter = -1
                        
                        events.append({
                            'type': 'note',
                            'step': step,
                            'octave': octave,
                            'alter': alter,
                            'duration': self.duration_map.get(duration, 24),
                            'duration_type': duration
                        })
                except (ValueError, IndexError, AttributeError):
                    continue
                    
            elif token.startswith('REST_'):
                # Parse rest: REST_8th
                try:
                    duration = token.split('_')[1]
                    events.append({
                        'type': 'rest',
                        'duration': self.duration_map.get(duration, 24),
                        'duration_type': duration
                    })
                except (ValueError, IndexError):
                    continue
        
        return events
    
    def create_musicxml(self, events: List[Dict[str, Any]], title: str = "AI Generated Music") -> str:
        """Convert musical events to MusicXML format"""
        
        # Create root element
        score = ET.Element('score-partwise', version="4.0")
        
        # Add work title
        work = ET.SubElement(score, 'work')
        work_title = ET.SubElement(work, 'work-title')
        work_title.text = title
        
        # Add identification
        identification = ET.SubElement(score, 'identification')
        creator = ET.SubElement(identification, 'creator', type="composer")
        creator.text = "Nathan Gendler"
        
        encoding = ET.SubElement(identification, 'encoding')
        software = ET.SubElement(encoding, 'software')
        software.text = "Bach AI Transformer"
        
        # Add defaults
        defaults = ET.SubElement(score, 'defaults')
        scaling = ET.SubElement(defaults, 'scaling')
        mm = ET.SubElement(scaling, 'millimeters')
        mm.text = "6.6"
        tenths = ET.SubElement(scaling, 'tenths')
        tenths.text = "40"
        
        # Add part list
        part_list = ET.SubElement(score, 'part-list')
        score_part = ET.SubElement(part_list, 'score-part', id="P1")
        part_name = ET.SubElement(score_part, 'part-name')
        part_name.text = "Piano"
        
        # Add MIDI instrument
        score_instrument = ET.SubElement(score_part, 'score-instrument', id="P1-I1")
        instrument_name = ET.SubElement(score_instrument, 'instrument-name')
        instrument_name.text = "Piano"
        
        # Create part
        part = ET.SubElement(score, 'part', id="P1")
        
        # Group events into measures (assume 4/4 time, 96 divisions per measure)
        measures = self.group_into_measures(events)
        
        for measure_num, measure_events in enumerate(measures, 1):
            measure = ET.SubElement(part, 'measure', number=str(measure_num))
            
            # Add attributes for first measure
            if measure_num == 1:
                attributes = ET.SubElement(measure, 'attributes')
                
                # Divisions
                divisions = ET.SubElement(attributes, 'divisions')
                divisions.text = str(self.divisions)
                
                # Key signature (find first key signature or use default)
                key_sig = next((e for e in events if e['type'] == 'key_signature'), None)
                if key_sig:
                    key = ET.SubElement(attributes, 'key')
                    fifths = ET.SubElement(key, 'fifths')
                    fifths.text = str(key_sig['fifths'])
                else:
                    key = ET.SubElement(attributes, 'key')
                    fifths = ET.SubElement(key, 'fifths')
                    fifths.text = str(self.default_key)
                
                # Time signature
                time_sig = next((e for e in events if e['type'] == 'time_signature'), None)
                if time_sig:
                    time = ET.SubElement(attributes, 'time')
                    beats = ET.SubElement(time, 'beats')
                    beats.text = str(time_sig['beats'])
                    beat_type = ET.SubElement(time, 'beat-type')
                    beat_type.text = str(time_sig['beat_type'])
                else:
                    time = ET.SubElement(attributes, 'time')
                    beats = ET.SubElement(time, 'beats')
                    beats.text = str(self.default_time_beats)
                    beat_type = ET.SubElement(time, 'beat-type')
                    beat_type.text = str(self.default_time_beat_type)
                
                # Clef
                clef = ET.SubElement(attributes, 'clef')
                sign = ET.SubElement(clef, 'sign')
                sign.text = "G"
                line = ET.SubElement(clef, 'line')
                line.text = "2"
            
            # Add notes and rests
            for event in measure_events:
                if event['type'] == 'note':
                    self.add_note_to_measure(measure, event)
                elif event['type'] == 'rest':
                    self.add_rest_to_measure(measure, event)
        
        return self.prettify_xml(score)
    
    def group_into_measures(self, events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group musical events into measures based on duration"""
        measures = []
        current_measure = []
        current_duration = 0
        measure_length = self.divisions * self.default_time_beats  # Full measure duration
        
        for event in events:
            if event['type'] in ['note', 'rest']:
                # If adding this event would exceed measure length, start new measure
                if current_duration + event['duration'] > measure_length and current_measure:
                    measures.append(current_measure)
                    current_measure = []
                    current_duration = 0
                
                current_measure.append(event)
                current_duration += event['duration']
                
                # If measure is exactly full, start new measure
                if current_duration >= measure_length:
                    measures.append(current_measure)
                    current_measure = []
                    current_duration = 0
        
        # Add any remaining events
        if current_measure:
            measures.append(current_measure)
        
        return measures
    
    def add_note_to_measure(self, measure: ET.Element, note_event: Dict[str, Any]):
        """Add a note element to a measure"""
        note = ET.SubElement(measure, 'note')
        
        # Pitch
        pitch = ET.SubElement(note, 'pitch')
        step = ET.SubElement(pitch, 'step')
        step.text = note_event['step']
        
        if note_event['alter'] != 0:
            alter = ET.SubElement(pitch, 'alter')
            alter.text = str(note_event['alter'])
        
        octave = ET.SubElement(pitch, 'octave')
        octave.text = str(note_event['octave'])
        
        # Duration
        duration = ET.SubElement(note, 'duration')
        duration.text = str(note_event['duration'])
        
        # Voice
        voice = ET.SubElement(note, 'voice')
        voice.text = "1"
        
        # Type
        note_type = ET.SubElement(note, 'type')
        type_map = {
            "16th": "16th",
            "8th": "eighth", 
            "quarter": "quarter",
            "half": "half",
            "whole": "whole"
        }
        note_type.text = type_map.get(note_event['duration_type'], "quarter")
        
        # Stem direction (basic logic)
        if note_event['octave'] >= 5:
            stem = ET.SubElement(note, 'stem')
            stem.text = "down"
        else:
            stem = ET.SubElement(note, 'stem')
            stem.text = "up"
    
    def add_rest_to_measure(self, measure: ET.Element, rest_event: Dict[str, Any]):
        """Add a rest element to a measure"""
        note = ET.SubElement(measure, 'note')
        
        # Rest
        rest = ET.SubElement(note, 'rest')
        
        # Duration
        duration = ET.SubElement(note, 'duration')
        duration.text = str(rest_event['duration'])
        
        # Voice
        voice = ET.SubElement(note, 'voice')
        voice.text = "1"
        
        # Type
        note_type = ET.SubElement(note, 'type')
        type_map = {
            "16th": "16th",
            "8th": "eighth", 
            "quarter": "quarter",
            "half": "half",
            "whole": "whole"
        }
        note_type.text = type_map.get(rest_event['duration_type'], "quarter")
    
    def prettify_xml(self, elem: ET.Element) -> str:
        """Return a pretty-printed XML string"""
        rough_string = ET.tostring(elem, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="  ")
        
        # Add XML declaration and DOCTYPE
        lines = pretty.split('\n')
        # Remove the auto-generated XML declaration
        if lines[0].startswith('<?xml'):
            lines = lines[1:]
        
        # Add proper declaration and DOCTYPE
        result = ['<?xml version="1.0" encoding="UTF-8"?>']
        result.append('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">')
        result.extend(line for line in lines if line.strip())
        
        return '\n'.join(result)
    
    def convert_file(self, input_file: str, output_file: str = None, title: str = None) -> str:
        """Convert a token file to MusicXML"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                tokens_text = f.read()
        except FileNotFoundError:
            return f"Error: Could not find file {input_file}"
        
        if not output_file:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}.xml"
        
        if not title:
            title = "Generated Bach"
        
        # Parse tokens and convert to MusicXML
        events = self.parse_tokens(tokens_text)
        
        if not events:
            return "Error: No valid musical events found in token file"
        
        musicxml = self.create_musicxml(events, title)
        
        # Save to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(musicxml)
            return f"Success: MusicXML saved to {output_file}"
        except Exception as e:
            return f"Error saving file: {e}"

def main():
    """Main function to convert tokens to MusicXML"""
    converter = TokensToMusicXMLConverter()
    
    # Look for your specific file first
    target_file = 'simple_generated_music.txt'
    possible_locations = [
        f'./Transformer/{target_file}',
        target_file,
        f'./{target_file}'
    ]
    
    print("🎼 Converting simple_generated_music.txt to MusicXML")
    print("=" * 50)
    
    # Find your specific file
    found_file = None
    for file_path in possible_locations:
        if os.path.exists(file_path):
            found_file = file_path
            print(f"✓ Found target file: {file_path}")
            break
    
    if not found_file:
        print(f"❌ Could not find {target_file}!")
        print("Checked these locations:")
        for location in possible_locations:
            print(f"  ✗ {location}")
        print(f"\nPlease make sure {target_file} exists in one of these locations.")
        return
    
    print(f"\n🎵 Converting {target_file} to MusicXML...")
    
    # Create output filename
    output_file = found_file.replace('.txt', '.xml')
    title = "Generated Bach"
    
    # Convert the file
    result = converter.convert_file(found_file, output_file, title)
    print(f"\nResult: {result}")
    
    if result.startswith("Success"):
        print(f"\n✅ Conversion successful!")
        print(f"📁 Input:  {found_file}")
        print(f"📁 Output: {output_file}")
        
        # Show file size and quick preview
        try:
            with open(found_file, 'r') as f:
                tokens = f.read().split()
            with open(output_file, 'r') as f:
                xml_content = f.read()
            
            print(f"\n📊 Conversion summary:")
            print(f"  🎵 Tokens processed: {len(tokens)}")
            print(f"  📄 MusicXML size: {len(xml_content):,} characters")
            print(f"  🎼 Musical events: {len([t for t in tokens if t.startswith(('NOTE_', 'REST_'))])}")
            
            # Show first few tokens for verification
            print(f"\n🎯 First 10 tokens converted:")
            for i, token in enumerate(tokens[:10], 1):
                print(f"  {i:2d}: {token}")
            if len(tokens) > 10:
                print(f"  ... and {len(tokens)-10} more")
            
        except Exception as e:
            print(f"  (Could not read file details: {e})")
        
        print(f"\n🎼 Next steps:")
        print(f"  1. Open {output_file} in MuseScore (free download)")
        print(f"  2. Listen to your AI Bach composition!")
        print(f"  3. Export as MP3/PDF if you like it")
        
    else:
        print(f"\n❌ Conversion failed!")
        print("Common issues:")
        print("  • File might be empty")
        print("  • Invalid token format")
        print("  • File permission issues")
        
        # Try to show file content for debugging
        try:
            with open(found_file, 'r') as f:
                content = f.read()
            print(f"\n🔍 File content preview (first 200 chars):")
            print(f"'{content[:200]}...'")
        except:
            print("  • Could not read file content")
    
    # Also check for other generated files
    print(f"\n🔍 Looking for other generated music files...")
    other_files = [
        './Transformer/generated_bach_50_tokens.txt',
        './Transformer/generated_bach_100_tokens.txt', 
        './Transformer/generated_bach_200_tokens.txt',
        './Transformer/bach_long_composition.txt',
        'generated_bach_100_tokens.txt'
    ]
    
    found_others = []
    for file_path in other_files:
        if os.path.exists(file_path) and file_path != found_file:
            found_others.append(file_path)
    
    if found_others:
        print(f"Found {len(found_others)} other generated music files:")
        for file_path in found_others:
            print(f"  ✓ {file_path}")
        print(f"\nTo convert these too, run:")
        print(f"converter = TokensToMusicXMLConverter()")
        for file_path in found_others:
            output_name = file_path.replace('.txt', '.xml')
            print(f"converter.convert_file('{file_path}', '{output_name}')")
    else:
        print("No other generated music files found.")

if __name__ == "__main__":
    main()
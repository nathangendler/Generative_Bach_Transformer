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
        
        part = ET.SubElement(score, 'part', id="P1")
        measures = self.group_into_measures(events)
        
        for measure_num, measure_events in enumerate(measures, 1):
            measure = ET.SubElement(part, 'measure', number=str(measure_num))
            
            if measure_num == 1:
                attributes = ET.SubElement(measure, 'attributes')
                
                # Divisions
                divisions = ET.SubElement(attributes, 'divisions')
                divisions.text = str(self.divisions)
                
                # Key signature
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
            
                clef = ET.SubElement(attributes, 'clef')
                sign = ET.SubElement(clef, 'sign')
                sign.text = "G"
                line = ET.SubElement(clef, 'line')
                line.text = "2"
            
            for event in measure_events:
                if event['type'] == 'note':
                    self.add_note_to_measure(measure, event)
                elif event['type'] == 'rest':
                    self.add_rest_to_measure(measure, event)
        
        return self.prettify_xml(score)
    
    def group_into_measures(self, events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        measures = []
        current_measure = []
        current_duration = 0
        measure_length = self.divisions * self.default_time_beats  # Full measure duration
        
        for event in events:
            if event['type'] in ['note', 'rest']:
                if current_duration + event['duration'] > measure_length and current_measure:
                    measures.append(current_measure)
                    current_measure = []
                    current_duration = 0
                
                current_measure.append(event)
                current_duration += event['duration']
                
                if current_duration >= measure_length:
                    measures.append(current_measure)
                    current_measure = []
                    current_duration = 0
        
        if current_measure:
            measures.append(current_measure)
        
        return measures
    
    def add_note_to_measure(self, measure: ET.Element, note_event: Dict[str, Any]):
        note = ET.SubElement(measure, 'note')
        pitch = ET.SubElement(note, 'pitch')
        step = ET.SubElement(pitch, 'step')
        step.text = note_event['step']
        
        if note_event['alter'] != 0:
            alter = ET.SubElement(pitch, 'alter')
            alter.text = str(note_event['alter'])
        octave = ET.SubElement(pitch, 'octave')
        octave.text = str(note_event['octave'])
        duration = ET.SubElement(note, 'duration')
        duration.text = str(note_event['duration'])
        voice = ET.SubElement(note, 'voice')
        voice.text = "1"
        note_type = ET.SubElement(note, 'type')
        type_map = {
            "16th": "16th",
            "8th": "eighth", 
            "quarter": "quarter",
            "half": "half",
            "whole": "whole"
        }
        note_type.text = type_map.get(note_event['duration_type'], "quarter")
        
        if note_event['octave'] >= 5:
            stem = ET.SubElement(note, 'stem')
            stem.text = "down"
        else:
            stem = ET.SubElement(note, 'stem')
            stem.text = "up"
    
    def add_rest_to_measure(self, measure: ET.Element, rest_event: Dict[str, Any]):
        note = ET.SubElement(measure, 'note')
        rest = ET.SubElement(note, 'rest')
        duration = ET.SubElement(note, 'duration')
        duration.text = str(rest_event['duration'])
        voice = ET.SubElement(note, 'voice')
        voice.text = "1"
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
        rough_string = ET.tostring(elem, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="  ")
        
        lines = pretty.split('\n')
        if lines[0].startswith('<?xml'):
            lines = lines[1:]
        
        result = ['<?xml version="1.0" encoding="UTF-8"?>']
        result.append('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">')
        result.extend(line for line in lines if line.strip())
        
        return '\n'.join(result)
    
    def convert_file(self, input_file, output_file, title) -> str:
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
        
        events = self.parse_tokens(tokens_text)
        
        if not events:
            return "Error: No valid musical events found in token file"
        
        musicxml = self.create_musicxml(events, title)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(musicxml)
            return f"Success: MusicXML saved to {output_file}"
        except Exception as e:
            return f"Error saving file: {e}"

def main():
    converter = TokensToMusicXMLConverter()
    
    target_file = 'generated_music.txt'
    possible_locations = [
        f'../generated/{target_file}',
        target_file,
        f'./{target_file}'
    ]
    
    found_file = None
    for file_path in possible_locations:
        if os.path.exists(file_path):
            found_file = file_path
            print(f"Found target file: {file_path}")
            break
    
    if not found_file:
        print(f"Could not find {target_file}")
        return
    print(f"Converting {target_file} to XML")
    output_file = found_file.replace('.txt', '.xml')
    title = "Generated Bach"
    result = converter.convert_file(found_file, output_file, title)
    print(f"\nResult: {result}")
    
    if result.startswith("Success"):
        print(f"Conversion successful") 
    else:
        print(f"Conversion failed")
        

if __name__ == "__main__":
    main()
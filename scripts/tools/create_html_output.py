#!/usr/bin/env python
"""Create an HTML version of the verse analysis for better visualization."""

import json
from pathlib import Path

def create_html_output():
    """Create HTML output from JSON analysis."""
    json_file = Path("data/verse_analysis_output.json")
    html_file = Path("data/verse_analysis_output.html")
    
    with open(json_file, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Verse Rhyme Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a1a;
            color: #e0e0e0;
        }
        h1 {
            color: #4CAF50;
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .verse {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            line-height: 2;
        }
        .line {
            margin: 10px 0;
            padding: 8px;
            background: #1e1e1e;
            border-left: 4px solid #4CAF50;
            padding-left: 15px;
        }
        .line-num {
            color: #888;
            font-weight: bold;
            margin-right: 10px;
        }
        .end-rhyme {
            background: #ff6b6b;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
        }
        .internal-exact {
            background: #4CAF50;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
        }
        .internal-slant {
            background: #ffa726;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
        }
        .internal-assonance {
            background: #42a5f5;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
        }
        .stats {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .stat-item {
            display: inline-block;
            margin: 10px 20px;
            padding: 10px;
            background: #1e1e1e;
            border-radius: 5px;
        }
        .stat-value {
            font-size: 2em;
            color: #4CAF50;
            font-weight: bold;
        }
        .stat-label {
            color: #888;
            font-size: 0.9em;
        }
        .details {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .detail-section {
            margin: 20px 0;
        }
        .detail-section h3 {
            color: #4CAF50;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 5px;
        }
        .rhyme-item {
            padding: 5px;
            margin: 5px 0;
            background: #1e1e1e;
            border-left: 3px solid #4CAF50;
            padding-left: 10px;
        }
    </style>
</head>
<body>
    <h1>Verse Rhyme Analysis</h1>
"""
    
    # Verse with highlights
    html += '<div class="verse">\n'
    for line_info in analysis["lines"]:
        line_num = line_info["line_num"]
        text = line_info["text"]
        words = line_info["words"]
        
        # Get end rhyme word
        end_rhyme_info = line_info.get("end_rhyme")
        end_word = words[-1].rstrip('.,!?;:') if words else ""
        
        # Get internal rhyme words
        internal_rhyme_map = {}
        if line_info.get("internal_rhymes"):
            for ir in line_info["internal_rhymes"]:
                word1 = ir.get("word1", "").lower()
                word2 = ir.get("word2", "").lower()
                rhyme_type = ir.get("type", "")
                internal_rhyme_map[word1] = rhyme_type
                internal_rhyme_map[word2] = rhyme_type
        
        # Build highlighted line
        html += f'<div class="line"><span class="line-num">{line_num:2d}</span>'
        
        for word in words:
            clean_word = word.rstrip('.,!?;:')
            is_end_rhyme = clean_word.lower() == end_word.lower() and end_rhyme_info
            is_internal_rhyme = clean_word.lower() in internal_rhyme_map
            
            if is_end_rhyme:
                rhyme_type = end_rhyme_info["rhyme_type"]
                html += f'<span class="end-rhyme" title="End rhyme: {rhyme_type}">{word}</span> '
            elif is_internal_rhyme:
                rhyme_type = internal_rhyme_map[clean_word.lower()]
                if rhyme_type == "EXACT":
                    html += f'<span class="internal-exact" title="Internal rhyme: {rhyme_type}">{word}</span> '
                elif rhyme_type == "SLANT":
                    html += f'<span class="internal-slant" title="Internal rhyme: {rhyme_type}">{word}</span> '
                else:
                    html += f'<span class="internal-assonance" title="Internal rhyme: {rhyme_type}">{word}</span> '
            else:
                html += f'{word} '
        
        html += '</div>\n'
    
    html += '</div>\n'
    
    # Statistics
    html += '<div class="stats">\n'
    html += '<h2>Statistics</h2>\n'
    html += f'<div class="stat-item"><div class="stat-value">{len(analysis["lines"])}</div><div class="stat-label">Total Lines</div></div>\n'
    html += f'<div class="stat-item"><div class="stat-value">{len(analysis["end_rhymes"])}</div><div class="stat-label">End Rhymes</div></div>\n'
    html += f'<div class="stat-item"><div class="stat-value">{len(analysis["internal_rhymes"])}</div><div class="stat-label">Internal Rhymes</div></div>\n'
    html += f'<div class="stat-item"><div class="stat-value">{analysis["rhyme_density"]:.1%}</div><div class="stat-label">Rhyme Density</div></div>\n'
    html += '</div>\n'
    
    # End rhymes
    if analysis["end_rhymes"]:
        html += '<div class="details">\n'
        html += '<div class="detail-section">\n'
        html += '<h3>End Rhymes</h3>\n'
        for er in analysis["end_rhymes"]:
            html += f'<div class="rhyme-item">Lines {er["line1"]}-{er["line2"]}: '
            html += f'<strong>{er["word1"]}</strong> / <strong>{er["word2"]}</strong> '
            html += f'({er["rhyme_type"]}, similarity: {er["similarity"]:.3f}, confidence: {er["confidence"]:.2f})</div>\n'
        html += '</div>\n'
        html += '</div>\n'
    
    # Internal rhymes summary
    if analysis["internal_rhymes"]:
        html += '<div class="details">\n'
        html += '<div class="detail-section">\n'
        html += '<h3>Internal Rhymes Summary</h3>\n'
        
        # Group by type
        type_groups = {}
        for ir in analysis["internal_rhymes"]:
            rt = ir.get("type", "UNKNOWN")
            if rt not in type_groups:
                type_groups[rt] = []
            word1 = ir.get("word1", "")
            word2 = ir.get("word2", "")
            type_groups[rt].append(f"{word1}/{word2}")
        
        for rhyme_type, words_list in sorted(type_groups.items()):
            html += f'<h4>{rhyme_type} ({len(words_list)})</h4>\n'
            html += '<div style="display: flex; flex-wrap: wrap; gap: 5px;">\n'
            for word_pair in words_list[:20]:  # Limit display
                html += f'<span style="background: #1e1e1e; padding: 3px 8px; border-radius: 3px;">{word_pair}</span>\n'
            if len(words_list) > 20:
                html += f'<span style="color: #888;">... and {len(words_list) - 20} more</span>\n'
            html += '</div>\n'
        
        html += '</div>\n'
        html += '</div>\n'
    
    html += """
    <div class="details">
        <div class="detail-section">
            <h3>Legend</h3>
            <p><span class="end-rhyme">End Rhyme</span> - Words at the end of consecutive lines that rhyme</p>
            <p><span class="internal-exact">Exact Internal Rhyme</span> - Perfect rhyme within a line</p>
            <p><span class="internal-slant">Slant Internal Rhyme</span> - Near rhyme within a line</p>
            <p><span class="internal-assonance">Assonance Internal Rhyme</span> - Vowel-only rhyme within a line</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"HTML output saved to: {html_file}")

if __name__ == "__main__":
    create_html_output()

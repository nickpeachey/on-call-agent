#!/usr/bin/env python3
"""
HTML generator for the AI On-Call Agent guide.
Creates a print-friendly HTML version that can be saved as PDF.
"""

import os
import sys
import webbrowser
from pathlib import Path

try:
    import markdown2
except ImportError:
    print("❌ Missing markdown2. Install with: pip install markdown2")
    sys.exit(1)

# CSS for print-friendly HTML
PRINT_CSS = """
<style>
@media print {
    body { margin: 0.5in; }
    h1 { page-break-before: always; }
    h1:first-child { page-break-before: avoid; }
    pre, blockquote { page-break-inside: avoid; }
    h1, h2, h3, h4, h5, h6 { page-break-after: avoid; }
}

@media screen {
    body { 
        max-width: 800px; 
        margin: 0 auto; 
        padding: 20px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }
}

body {
    line-height: 1.6;
    color: #333;
}

h1 {
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
    font-size: 24px;
}

h2 {
    color: #34495e;
    border-bottom: 2px solid #ecf0f1;
    padding-bottom: 5px;
    margin-top: 30px;
    font-size: 20px;
}

h3 {
    color: #2980b9;
    margin-top: 25px;
    font-size: 16px;
}

h4 {
    color: #8e44ad;
    margin-top: 20px;
    font-size: 14px;
}

code {
    background-color: #f8f9fa;
    padding: 2px 4px;
    border-radius: 3px;
    font-family: 'Monaco', 'Consolas', monospace;
    font-size: 12px;
}

pre {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #3498db;
    overflow-x: auto;
    font-family: 'Monaco', 'Consolas', monospace;
    font-size: 11px;
    line-height: 1.4;
}

pre code {
    background: none;
    padding: 0;
}

blockquote {
    border-left: 4px solid #3498db;
    margin: 0;
    padding: 10px 20px;
    background-color: #ecf0f1;
    font-style: italic;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-size: 12px;
}

th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}

th {
    background-color: #3498db;
    color: white;
    font-weight: bold;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

ul, ol {
    margin: 10px 0;
    padding-left: 25px;
}

li {
    margin: 5px 0;
}

.print-instructions {
    background-color: #d1ecf1;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #17a2b8;
    margin: 20px 0;
}

@media print {
    .print-instructions { display: none; }
}
</style>
"""

def convert_markdown_to_html(markdown_file: Path, output_file: Path):
    """Convert markdown file to print-friendly HTML."""
    
    print(f"📖 Reading markdown file: {markdown_file}")
    
    # Read the markdown content
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    print("🔄 Converting markdown to HTML...")
    
    # Convert markdown to HTML with extensions
    html_content = markdown2.markdown(
        markdown_content,
        extras=[
            'fenced-code-blocks',
            'tables',
            'header-ids',
            'toc',
            'strike',
            'task_list'
        ]
    )
    
    # Create print instructions
    print_instructions = """
    <div class="print-instructions">
        <h3>📄 How to Save as PDF</h3>
        <p><strong>On macOS:</strong></p>
        <ol>
            <li>Press <kbd>Cmd + P</kbd> to open print dialog</li>
            <li>Click the <strong>PDF</strong> dropdown in bottom-left</li>
            <li>Select <strong>"Save as PDF..."</strong></li>
            <li>Choose location and filename</li>
            <li>Click <strong>Save</strong></li>
        </ol>
        <p><strong>On Windows/Linux:</strong></p>
        <ol>
            <li>Press <kbd>Ctrl + P</kbd> to open print dialog</li>
            <li>Select <strong>"Save as PDF"</strong> as destination</li>
            <li>Click <strong>Save</strong></li>
        </ol>
        <p><em>This instruction box will not appear in the printed version.</em></p>
    </div>
    """
    
    # Wrap in complete HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>AI On-Call Agent - Complete Dummy's Guide</title>
        {PRINT_CSS}
    </head>
    <body>
        {print_instructions}
        <div class="content">
            {html_content}
        </div>
    </body>
    </html>
    """
    
    print("💾 Saving HTML file...")
    
    # Save HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    # Get file size for reporting
    file_size = output_file.stat().st_size / 1024  # KB
    
    print(f"✅ HTML generated successfully!")
    print(f"📁 Output file: {output_file}")
    print(f"📊 File size: {file_size:.1f} KB")
    
    return True

def main():
    """Main function."""
    
    # Paths
    current_dir = Path(__file__).parent
    markdown_file = current_dir / "AI_ON_CALL_AGENT_DUMMYS_GUIDE.md"
    output_file = current_dir / "AI_ON_CALL_AGENT_DUMMYS_GUIDE.html"
    
    print("🤖 AI On-Call Agent - HTML Generator")
    print("=" * 50)
    
    # Check if markdown file exists
    if not markdown_file.exists():
        print(f"❌ Markdown file not found: {markdown_file}")
        sys.exit(1)
    
    # Convert to HTML
    success = convert_markdown_to_html(markdown_file, output_file)
    
    if success:
        print("\n🎉 HTML generation complete!")
        print("\n📖 Next steps:")
        print(f"   1. Opening HTML file in browser...")
        print(f"   2. Use Cmd+P (Mac) or Ctrl+P (Windows/Linux) to print")
        print(f"   3. Select 'Save as PDF' in the print dialog")
        print(f"   4. Choose filename and location")
        
        # Try to open the HTML file automatically
        try:
            webbrowser.open(f"file://{output_file.absolute()}")
            print("   ✅ HTML opened in browser!")
        except Exception as e:
            print(f"   ⚠️ Could not open browser automatically: {e}")
            print(f"   📂 Manually open: {output_file}")
            
    else:
        print("\n💥 HTML generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

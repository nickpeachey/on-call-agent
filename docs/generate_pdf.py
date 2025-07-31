#!/usr/bin/env python3
"""
Script to convert the AI On-Call Agent Dummy's Guide to PDF format.
Requires: pip install markdown2 weasyprint
"""

import os
import sys
from pathlib import Path

try:
    import markdown2
    from weasyprint import HTML, CSS
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("📦 Install with: pip install markdown2 weasyprint")
    sys.exit(1)

# CSS styles for PDF formatting
PDF_CSS = """
@page {
    size: A4;
    margin: 2cm 1.5cm;
    counter-increment: page;
    @bottom-center {
        content: "AI On-Call Agent Guide - Page " counter(page);
        font-size: 10px;
        color: #666;
    }
}

body {
    font-family: 'Arial', 'Helvetica', sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: none;
}

h1 {
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
    page-break-before: always;
    font-size: 24px;
}

h1:first-child {
    page-break-before: auto;
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

.highlight {
    background-color: #fff3cd;
    padding: 10px;
    border-radius: 5px;
    border-left: 4px solid #ffc107;
    margin: 15px 0;
}

.note {
    background-color: #d1ecf1;
    padding: 10px;
    border-radius: 5px;
    border-left: 4px solid #17a2b8;
    margin: 15px 0;
}

.warning {
    background-color: #f8d7da;
    padding: 10px;
    border-radius: 5px;
    border-left: 4px solid #dc3545;
    margin: 15px 0;
}

ul, ol {
    margin: 10px 0;
    padding-left: 25px;
}

li {
    margin: 5px 0;
}

.toc {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 5px;
    margin: 20px 0;
}

.toc ul {
    list-style-type: none;
    padding-left: 15px;
}

.toc > ul {
    padding-left: 0;
}

.toc a {
    text-decoration: none;
    color: #2980b9;
}

.toc a:hover {
    text-decoration: underline;
}

/* Page breaks */
.page-break {
    page-break-before: always;
}

/* Prevent breaks inside code blocks */
pre, blockquote {
    page-break-inside: avoid;
}

/* Keep headings with following content */
h1, h2, h3, h4, h5, h6 {
    page-break-after: avoid;
}
"""

def convert_markdown_to_pdf(markdown_file: Path, output_file: Path):
    """Convert markdown file to PDF."""
    
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
    
    # Wrap in complete HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>AI On-Call Agent - Complete Dummy's Guide</title>
        <style>{PDF_CSS}</style>
    </head>
    <body>
        <div class="content">
            {html_content}
        </div>
    </body>
    </html>
    """
    
    print("📄 Generating PDF...")
    
    # Convert HTML to PDF
    try:
        html_doc = HTML(string=full_html)
        html_doc.write_pdf(output_file)
        
        # Get file size for reporting
        file_size = output_file.stat().st_size / 1024 / 1024  # MB
        
        print(f"✅ PDF generated successfully!")
        print(f"📁 Output file: {output_file}")
        print(f"📊 File size: {file_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return False

def main():
    """Main function."""
    
    # Paths
    current_dir = Path(__file__).parent
    markdown_file = current_dir / "AI_ON_CALL_AGENT_DUMMYS_GUIDE.md"
    output_file = current_dir / "AI_ON_CALL_AGENT_DUMMYS_GUIDE.pdf"
    
    print("🤖 AI On-Call Agent - PDF Generator")
    print("=" * 50)
    
    # Check if markdown file exists
    if not markdown_file.exists():
        print(f"❌ Markdown file not found: {markdown_file}")
        sys.exit(1)
    
    # Convert to PDF
    success = convert_markdown_to_pdf(markdown_file, output_file)
    
    if success:
        print("\n🎉 PDF generation complete!")
        print("\n📖 You can now:")
        print(f"   • Open the PDF: open '{output_file}'")
        print(f"   • Share the file: {output_file}")
        print(f"   • Print physical copies for your team")
    else:
        print("\n💥 PDF generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

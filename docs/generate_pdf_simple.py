#!/usr/bin/env python3
"""
Simple PDF generator using pandoc (more reliable on macOS).
Install with: brew install pandoc
"""

import os
import sys
import subprocess
from pathlib import Path

def check_pandoc():
    """Check if pandoc is installed."""
    try:
        result = subprocess.run(['pandoc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Pandoc found")
            return True
        else:
            return False
    except FileNotFoundError:
        return False

def generate_pdf_with_pandoc(markdown_file: Path, output_file: Path):
    """Generate PDF using pandoc."""
    
    # Pandoc command with good formatting options
    cmd = [
        'pandoc',
        str(markdown_file),
        '-o', str(output_file),
        '--pdf-engine=wkhtmltopdf',
        '--toc',  # Table of contents
        '--toc-depth=3',
        '-V', 'geometry:margin=1in',
        '-V', 'fontsize=11pt',
        '--highlight-style=github'
    ]
    
    try:
        print("📄 Generating PDF with pandoc...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            file_size = output_file.stat().st_size / 1024 / 1024  # MB
            print(f"✅ PDF generated successfully!")
            print(f"📁 Output file: {output_file}")
            print(f"📊 File size: {file_size:.1f} MB")
            return True
        else:
            print(f"❌ Pandoc error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running pandoc: {e}")
        return False

def generate_pdf_simple(markdown_file: Path, output_file: Path):
    """Generate PDF using basic pandoc without external dependencies."""
    
    # Simpler pandoc command that doesn't require wkhtmltopdf
    cmd = [
        'pandoc',
        str(markdown_file),
        '-o', str(output_file),
        '--toc',  # Table of contents
        '--toc-depth=3',
        '-V', 'geometry:margin=1in',
        '-V', 'fontsize=11pt'
    ]
    
    try:
        print("📄 Generating PDF with pandoc (basic)...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            file_size = output_file.stat().st_size / 1024 / 1024  # MB
            print(f"✅ PDF generated successfully!")
            print(f"📁 Output file: {output_file}")
            print(f"📊 File size: {file_size:.1f} MB")
            return True
        else:
            print(f"❌ Pandoc error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running pandoc: {e}")
        return False

def main():
    """Main function."""
    
    # Paths
    current_dir = Path(__file__).parent
    markdown_file = current_dir / "AI_ON_CALL_AGENT_DUMMYS_GUIDE.md"
    output_file = current_dir / "AI_ON_CALL_AGENT_DUMMYS_GUIDE.pdf"
    
    print("🤖 AI On-Call Agent - Simple PDF Generator")
    print("=" * 50)
    
    # Check if markdown file exists
    if not markdown_file.exists():
        print(f"❌ Markdown file not found: {markdown_file}")
        sys.exit(1)
    
    # Check if pandoc is available
    if not check_pandoc():
        print("❌ Pandoc not found!")
        print("📦 Install with: brew install pandoc")
        print("   Or visit: https://pandoc.org/installing.html")
        sys.exit(1)
    
    # Try to generate PDF
    success = generate_pdf_simple(markdown_file, output_file)
    
    if success:
        print("\n🎉 PDF generation complete!")
        print("\n📖 You can now:")
        print(f"   • Open the PDF: open '{output_file}'")
        print(f"   • Share the file: {output_file}")
        print(f"   • Print physical copies for your team")
        
        # Try to open the PDF automatically
        try:
            subprocess.run(['open', str(output_file)], check=False)
            print("   • PDF opened automatically! 📖")
        except:
            pass
            
    else:
        print("\n💥 PDF generation failed!")
        print("📖 You can still read the markdown version:")
        print(f"   • Markdown file: {markdown_file}")
        sys.exit(1)

if __name__ == "__main__":
    main()

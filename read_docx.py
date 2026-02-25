import sys
try:
    import docx
    doc = docx.Document(sys.argv[1])
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    print("\n".join(text)[:3000]) # Print first 3000 chars
except ImportError:
    print("python-docx not installed")
except Exception as e:
    print(f"Error: {e}")

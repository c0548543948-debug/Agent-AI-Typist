"""
יוצר docx מאפס ב-XML גולמי — ללא python-docx.
כדי לבדוק אם הבעיה בספרייה או ב-Word עצמו.
"""
import zipfile, io

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>"""

DOCUMENT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr>
        <w:bidi/>
        <w:jc w:val="right"/>
      </w:pPr>
      <w:r>
        <w:rPr>
          <w:rtl/>
          <w:lang w:bidi="he-IL"/>
        </w:rPr>
        <w:t>שלום זה צריך להיות מיושר לימין</w:t>
      </w:r>
    </w:p>
    <w:p>
      <w:r>
        <w:t>This should be left-aligned (LTR)</w:t>
      </w:r>
    </w:p>
    <w:sectPr>
      <w:bidi/>
    </w:sectPr>
  </w:body>
</w:document>"""

SETTINGS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:bidi/>
  <w:compat>
    <w:compatSetting w:name="compatibilityMode"
      w:uri="http://schemas.microsoft.com/office/word"
      w:val="15"/>
  </w:compat>
</w:settings>"""

STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:pPrDefault>
      <w:pPr>
        <w:bidi/>
        <w:jc w:val="right"/>
      </w:pPr>
    </w:pPrDefault>
    <w:rPrDefault>
      <w:rPr>
        <w:lang w:bidi="he-IL"/>
      </w:rPr>
    </w:rPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:pPr>
      <w:bidi/>
      <w:jc w:val="right"/>
    </w:pPr>
  </w:style>
</w:styles>"""

buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.writestr('[Content_Types].xml', CONTENT_TYPES.encode('utf-8'))
    zf.writestr('_rels/.rels', RELS.encode('utf-8'))
    zf.writestr('word/_rels/document.xml.rels', DOC_RELS.encode('utf-8'))
    zf.writestr('word/document.xml', DOCUMENT.encode('utf-8'))
    zf.writestr('word/settings.xml', SETTINGS.encode('utf-8'))
    zf.writestr('word/styles.xml', STYLES.encode('utf-8'))

with open('test_rtl_raw.docx', 'wb') as f:
    f.write(buf.getvalue())

print("נשמר: test_rtl_raw.docx")
print("פתח אותו ב-Word ובדוק אם הטקסט העברי מיושר לימין")

"""Conservative snippet-candidate pattern filter; design/test only."""
import re

PATTERNS={
 'email': re.compile(r'(?<![\w.+-])[\w.!#$%&\x27*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+',re.I),
 'resident_registration': re.compile(r'(?<!\d)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\s*[-–]?\s*[1-8]\d{6}(?!\d)'),
 'phone': re.compile(r'(?<!\d)(?:(?:\+82[ .-]?(?:0)?|0)(?:1[016789]|2|[3-6]\d|70)[ .-]?\d{3,4}[ .-]?\d{4}|(?:\+1[ .-]?)?\(?[2-9]\d{2}\)?[ .-]\d{3}[ .-]\d{4})(?!\d)'),
 'account': re.compile(r'(?:계좌(?:번호)?|통장|account(?:\s+(?:no\.?|number))?|IBAN)\s*[:：=]?\s*(?:[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}|\d(?:[\d -]{5,24})\d)|(?<!\w)[A-Z]{2}\d{2}[A-Z0-9]{11,30}(?!\w)',re.I),
 'address': re.compile(r'(?:[가-힣0-9·]+(?:대로|로|길)\s*\d+(?:[-–]\d+)?(?:\s+\d+동\s*\d+호)?|[가-힣]+(?:동|읍|면|리)\s+\d+(?:-\d+)?|\b\d{1,6}\s+(?:[A-Za-z0-9.\x27-]+\s+){0,6}(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Lane|Ln\.?|Drive|Dr\.?|Boulevard|Blvd\.?|Court|Ct\.?)\b)',re.I),
}

def categories(text):return sorted(name for name,pattern in PATTERNS.items() if pattern.search(text))
def should_drop(text):return bool(categories(text))


FRAGMENT=re.compile(r'(?<![0-9])[-–—.]?[0-9]+(?:[-–—.][0-9]+)+')
def fragment(text):
    return any(m.start()<32 and sum(c.isascii() and c.isdigit() for c in m.group())>=7 for m in FRAGMENT.finditer(text))
def boundary_hits(previous,current):
    tail=previous[-256:];head=current[:256];joined=tail+head;b=len(tail);hits=[];cross=False
    for name,p in PATTERNS.items():
        for m in p.finditer(joined):
            if m.end()>b:hits.append('boundary_'+name);cross|=m.start()<b
    if fragment(current):hits.append('head_digit_fragment')
    return sorted(set(hits)),cross

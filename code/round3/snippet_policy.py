"""Fixed lexical concealment only; not a semantic toxicity classifier."""
import re
WINDOW_BEFORE=15
WINDOW_AFTER=16
ENGLISH=['fuck','shit','bitch','cunt','motherfucker','nigger','faggot','retard']
KOREAN=['\uc2dc\ubc1c','\uc528\ubc1c','\uc881','\ubcd1\uc2e0','\uac1c\uc0c8\ub07c','\uc9c0\ub784','\ud55c\ub0a8\ucda9','\uae40\uce58\ub140']
OFFENSE=re.compile('|'.join([r'\b'+re.escape(x)+r'\b' for x in ENGLISH]+[re.escape(x) for x in KOREAN]),re.I)

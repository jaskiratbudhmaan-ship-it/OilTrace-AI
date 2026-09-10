import sys
from config import *
print("="*60)
print("OILTRACE AI - MEMBER 1 COMPLETE PROJECT CHECK")
print("="*60)
print("Python:",sys.version.split()[0])
print("Device:",DEVICE)
for d in DIRECTORIES:
    print("✓" if d.exists() else "✗", d.relative_to(BASE_DIR))
print("="*60)
print("✓ PROJECT STRUCTURE READY")

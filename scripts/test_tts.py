#!/usr/bin/env python3
"""Check text-to-speech: Google Cloud (service account) first, then the edge-tts fallback.  python scripts/test_tts.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from playq import config  # loads .env
from playq.agents.prototype import _cloud_token, _cloud_voices, synthesize_dialogue

print("1) service account:", end=" ")
path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
if not path:
    print("GOOGLE_APPLICATION_CREDENTIALS is empty in .env -> Cloud TTS unused (edge-tts will be the voice)")
elif not os.path.exists(os.path.expanduser(path)):
    print(f"file not found: {path}")
else:
    try:
        tok = _cloud_token(); print(f"ok (project {tok[1]})")
        print("2) Hindi voices:", _cloud_voices(*tok))
    except Exception as e:
        print("FAILED:", str(e)[:300])
        print("   Typical fixes: enable 'Cloud Text-to-Speech API' in the project; give the service account the 'Service Usage Consumer' role.")
lines = [{"start_s": 0.3, "speaker": "A", "text_roman": "Arre wah, kya shot hai!", "speech": "अरे वाह, क्या shot है!", "emotion": "x", "segment": 0},
         {"start_s": 3.0, "speaker": "B", "text_roman": "Kitchen pe dink ka jaadu!", "speech": "kitchen पे dink का जादू!", "emotion": "x", "segment": 0}]
out = os.path.join(tempfile.mkdtemp(), "t.wav")
res = synthesize_dialogue(lines, 7.0, out)
print("3) result:", {k: v for k, v in res.items() if k != "timing"})
print("   wav:", out if os.path.exists(out) else "not created")

import asyncio
import time
import os
import sys

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.main import app, lifespan, health, languages, translate_text
from api.models import TranslationRequest

def run_api_tests():
    print("=" * 60)
    print("Testing FastAPI REST Endpoint Plumbing")
    print("=" * 60)

    async def test():
        async with lifespan(app):
            # 1. Health check
            print("\n[1] Testing GET /health ...")
            h_resp = health()
            print("Status   :", h_resp.status)
            print("Model loaded:", h_resp.model_loaded)
            print("Device   :", h_resp.device)
            assert h_resp.model_loaded is True, "Model not loaded"

            # 2. Languages check
            print("\n[2] Testing GET /languages ...")
            l_resp = languages()
            print("Supported pairs:", l_resp.supported_pairs)

            # 3. Translation check
            print("\n[3] Testing POST /translate ...")
            req = TranslationRequest(
                text="አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
                source_lang="am",
                target_lang="en"
            )
            print(f"Request: text='{req.text}'")
            start = time.perf_counter()
            t_resp = translate_text(req)
            elapsed_ms = (time.perf_counter() - start) * 1000

            print(f"Translation     : {t_resp.translation}")
            print(f"Compute time    : {t_resp.compute_time_ms} ms (Total: {elapsed_ms:.1f} ms)")

            assert len(t_resp.translation) > 0, "Translation string is empty"

            print("\n" + "=" * 60)
            print("ALL FASTAPI REST ENDPOINT TESTS PASSED SUCCESSFULLY! ✅")
            print("=" * 60)

    asyncio.run(test())

if __name__ == "__main__":
    run_api_tests()

"""
Image-generation skill. Lets the user ask Cleo to "draw / generate / make a
picture of ..." and have a Stable Diffusion image rendered inline in the chat
window.

The model calls generate_image with a prompt; we run it through the isolated
image helper (modules/imagegen.py), which saves a PNG and registers it as the
turn's pending image. assistant.py then attaches it to the logged reply as an
[[IMAGE:name]] marker, which the control panel renders as an <img>. We return
only a short status to the model (no base64) so its context stays small and it
answers conversationally rather than echoing image data.
"""
import config
from modules.skills import skill


@skill(
    "generate_image",
    "Generate/create/draw an image or picture from a text description. Use this "
    "whenever the user asks you to draw, generate, create, paint, or make an "
    "image, picture, or artwork of something. The image is shown to the user "
    "automatically.",
    {"prompt": "a detailed description of the image to generate, e.g. "
               "'a watercolor fox sitting in a snowy forest at dusk'"},
)
def _generate_image(args: dict) -> dict:
    if not config.IMAGEGEN_ENABLED:
        return {"error": "image generation is turned off"}
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return {"error": "no prompt was provided"}
    from modules import imagegen
    try:
        imagegen.generate_to_file(prompt)
    except Exception as e:
        return {"error": f"image generation failed: {e}"}
    # The PNG is attached to the reply by assistant.py; the model just needs to
    # acknowledge it briefly.
    return {"status": "image generated and displayed to the user",
            "prompt": prompt}

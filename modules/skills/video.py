"""
Video-generation skill. Lets the user ask Cleo to "make a video of ..." and have
a short generated clip rendered inline in the chat window.

Mirrors the image skill: the model calls generate_video with a prompt; we run it
through the isolated video helper (modules/videogen.py), which saves an MP4 and
registers it as the turn's pending video. assistant.py then attaches it to the
logged reply as a [[VIDEO:name]] marker, which the control panel renders as a
<video>. We return only a short status to the model (no binary data).
"""
import config
from modules.skills import skill


@skill(
    "generate_video",
    "Generate/create a short video or animation clip from a text description. Use "
    "this when the user asks you to make, generate, or create a video, clip, gif, "
    "or animation of something. The video is shown to the user automatically. Note "
    "it can take a while to render.",
    {"prompt": "a detailed description of the video to generate, e.g. "
               "'a fox running through a snowy forest, cinematic, slow motion'"},
)
def _generate_video(args: dict) -> dict:
    if not config.VIDEOGEN_ENABLED:
        return {"error": "video generation is turned off"}
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return {"error": "no prompt was provided"}
    from modules import videogen
    try:
        videogen.generate_to_file(prompt)
    except Exception as e:
        return {"error": f"video generation failed: {e}"}
    return {"status": "video generated and displayed to the user",
            "prompt": prompt}

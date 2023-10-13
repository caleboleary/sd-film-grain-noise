import modules.scripts as scripts
import gradio as gr
import os
import numpy as np
import cv2

from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):  

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "Add Film Grain"


# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):

        return True

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        intensity_slider = gr.Slider(
            minimum=1,
            maximum=100,
            step=1,
            value=7,
            label="Intensity"
        )
        return [intensity_slider]

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p, intensity_slider):

        def add_film_grain(img, alpha=0.07):
            from PIL import Image

            # Convert PIL Image to OpenCV format
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            h, w, c = img.shape
            grain_mono = np.random.normal(0, 50, (h, w)).astype(np.float32)
            grain = np.stack([grain_mono]*c, axis=2)

            grain = 2 * (grain - np.min(grain)) / (np.max(grain) - np.min(grain)) - 1
            img_with_grain = (1 - alpha) * img + alpha * grain * 255
            img_with_grain = np.clip(img_with_grain, 0, 255).astype(np.uint8)

            # Convert back to PIL Image
            img_with_grain = cv2.cvtColor(img_with_grain, cv2.COLOR_BGR2RGB)
            img_with_grain_pil = Image.fromarray(img_with_grain.astype('uint8'), 'RGB')
            
            return img_with_grain_pil


        # If overwrite is false, append the rotation information to the filename
        # using the "basename" parameter and save it in the same directory.
        # If overwrite is true, stop the model from saving its outputs and
        # save the rotated and flipped images instead.
        basename = ""
        p.do_not_save_samples = True

        proc = process_images(p)

        # rotate and flip each image in the processed images
        # use the save_images method from images.py to save
        # them.
        for i in range(len(proc.images)):

            proc.images[i] = add_film_grain(proc.images[i], intensity_slider / 100.0)

            images.save_image(proc.images[i], p.outpath_samples, basename,
            proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p)

        return proc
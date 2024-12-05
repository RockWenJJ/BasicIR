import transformers
from transformers import CLIPProcessor, CLIPModel
import torch
import os
from tqdm import tqdm
from PIL import Image
import random

from basicir.utils.misc import restore_single_img, save_img

def random_crop(image, crop_size=(224, 224)):
    width, height = image.size
    if width > crop_size[0] and height > crop_size[1]:
        x = random.randint(0, width - crop_size[0])
        y = random.randint(0, height - crop_size[1])
        return image.crop((x, y, x + crop_size[0], y + crop_size[1]))
    else:
        return image

class Rater:
    """Base rater for single image restoration with VLM rating."""
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rate_freq = opt['rate_freq']
        self.rating_model = CLIPModel.from_pretrained(opt['rating_model'])
        self.rating_processor = CLIPProcessor.from_pretrained(opt['rating_model'])
        self.rating_model.to(self.device)

        self.db_path = opt['db_path']
        self.db_files = os.listdir(self.db_path)
        self.text_prompts = ["This water quality of the scene is bad", 
                             "The water quality of the scene is poor", 
                             "The water quality of the scene is fair", 
                             "The water quality of the scene is good", 
                             "The water quality of the scene is excellent"]
        self.mini_batch_size = 4
    
    def rate_and_update_database(self, model):
        """Rate and update database."""
        print("Rating images...")
        ave_origin_score = 0
        ave_restored_score = 0
        update_count = 0
        for image in tqdm(self.db_files, desc="Rating images", total=len(self.db_files)):
            # rate origin image
            origin_img_path = os.path.join(self.db_path, image)
            origin_img = Image.open(origin_img_path)
            origin_score = self.rate_image(origin_img)
            # rate restored image
            restored_img = restore_single_img(model, origin_img_path)
            restored_score = self.rate_image(Image.fromarray(restored_img))
            # update average score
            ave_origin_score += origin_score
            ave_restored_score += restored_score
            if restored_score > origin_score:
                # update database
                save_img(origin_img_path, restored_img)
                update_count += 1

        print(f"Average origin score: {ave_origin_score / len(self.db_files):.2f}")
        print(f"Average restored score: {ave_restored_score / len(self.db_files):.2f}")
        print(f"Update ratio: {update_count / len(self.db_files)*100:.2f}%")

    
    def rate_image(self, img):
        images = [random_crop(img) for _ in range(self.mini_batch_size)]
        inputs = self.rating_processor(images=images, text=self.text_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.rating_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            score = probs[0:self.mini_batch_size,0] * 1 + probs[0:self.mini_batch_size,1] * 2 + probs[0:self.mini_batch_size,2] * 3 + probs[0:self.mini_batch_size,3] * 4 + probs[0:self.mini_batch_size,4] * 5
            score = score.mean().item()
        return score




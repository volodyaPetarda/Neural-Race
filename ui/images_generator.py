import copy
from typing import List

import pygame
import torch
from diffusers import UNet2DModel, DDPMScheduler
from torchvision.transforms import transforms, InterpolationMode

from ui.button import Button
from ui.progress_bar import ProgressBar
from ui.text_input import TextInput


class ImageGenerator:
    def __init__(self, screen):
        device = 'cuda'
        saved_model = torch.load("data/models/model_15.pth", map_location=device)

        self.screen = screen
        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=3,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        ).to(device)
        self.model.load_state_dict(saved_model['model_state_dict'])
        self.generator_steps = 1000
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.generator_steps,
            beta_schedule="squaredcos_cap_v2",
        )

        self.image_generator = self.generate_images(5)
        self.tensor_to_image = transforms.Compose([
            transforms.Lambda(lambda x: (x * 0.5 + 0.5).clamp(0, 1).cpu()),
            transforms.ToPILImage(),
        ])
        self.step = 0
        self.positions = [(256 + i * (256 + 32), 300) for i in range(5)]
        self._exited = False
        self.images = None

        self.progress_bar = ProgressBar(200, 1000, 1720, 900, self.generator_steps, self.screen)

        restart_button_image = pygame.image.load("data/images/build_trace.png")
        restart_button_pressed = pygame.image.load("data/images/build_trace_pressed.png")
        self.restart_button = Button(300, 650, restart_button_image, restart_button_pressed, 600, 200, self.screen)

        start_button_image = pygame.image.load("data/images/start_trace.png")
        start_button_pressed = pygame.image.load("data/images/start_trace_pressed.png")
        self.start_button = Button(1020, 650, start_button_image, start_button_pressed, 600, 200, self.screen)

        self.trace_name_input = TextInput(
            400, 50, 1180, 150, pygame.font.Font(None, 128), self.screen
        )
        self.images_generated = False

    def next_frame(self, delta_time: float, events: List[pygame.event.Event]):
        self.restart_button.next_frame(events)
        self.start_button.next_frame(events)
        self.trace_name_input.next_frame(delta_time, events)

        if self.restart_button.clicked:
            self.images_generated = False
            self.progress_bar.current_step = 0
            self.image_generator = self.generate_images(5)

        if self.start_button.clicked and self.images_generated:
            self._exited = True

        try:
            images = next(self.image_generator)
        except StopIteration:
            self.images_generated = True
        else:
            self.progress_bar.step()
            self.step += 1
            self.images = list(map(self.tensor_to_image, images))
        self.progress_bar.next_frame()
        self.draw_images()



    def draw_images(self):
        images = self.get_images(256)
        for image, position in zip(images, self.positions):
            self.screen.blit(image, position)

    def pil_image_to_surface(self, pil_image):
        return pygame.image.fromstring(
            pil_image.tobytes(), pil_image.size, pil_image.mode).convert()

    @torch.no_grad()
    def generate_images(self, batch_size=1):
        self.model.eval()
        sample_size = self.model.config.sample_size
        in_channels = self.model.config.in_channels

        images = torch.randn(batch_size, in_channels, sample_size, sample_size).to(self.model.device)
        for t in self.noise_scheduler.timesteps:
            timesteps = t.unsqueeze(0).to(self.model.device)
            noise_pred = self.model(images, timesteps).sample

            scheduler_output = self.noise_scheduler.step(noise_pred, t, images)
            images = scheduler_output.prev_sample.detach()
            yield images

    @property
    def exited(self):
        return self._exited

    def get_images(self, size: int):
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=InterpolationMode.NEAREST),
        ])
        images = list(map(transform, self.images))
        images = list(map(self.pil_image_to_surface, images))
        return images
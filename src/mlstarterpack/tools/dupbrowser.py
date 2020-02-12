"""Duplicate images browser for dupfinder.py"""
from dataclasses import dataclass, field
from pathlib import Path
import tkinter as tk
from typing import *

import pandas as pd
from PIL import ImageTk, Image

from mlstarterpack.vision.images import load_img


@dataclass
class ImageBlacklist:
    blacklist_path: Union[str, Path]
    blacklist: Set[str] = field(default_factory=set)

    def __str__(self):
        return str(self.blacklist)

    def __len__(self):
        return len(self.blacklist)

    def __contains__(self, item):
        return item in self.blacklist

    def __bool__(self):
        return bool(self.blacklist)

    def add(self, image_name: str):
        self.blacklist.add(image_name)

    def save(self):
        blacklist_series = pd.Series(list(self.blacklist))
        blacklist_series.to_csv(self.blacklist_path, index=False)

    @classmethod
    def load(cls, blacklist_path: Union[str, Path]):
        blacklist = pd.read_csv(blacklist_path, header=None, names=['image_name'])
        return cls(blacklist_path, set(blacklist.image_name))


@dataclass
class ImagePair:
    image1_path: Path
    image2_path: Path
    distance: float


class DuplicateBrowser:  # pylint: disable=too-many-instance-attributes
    def __init__(self, image_pairs: List[ImagePair], data_root: Path, output_path: str):
        self.data_root = data_root

        blacklist_path = Path(output_path)
        if blacklist_path.exists():
            self.blacklist = ImageBlacklist.load(blacklist_path)
        else:
            self.blacklist = ImageBlacklist(blacklist_path=output_path)

        if self.blacklist:
            print(f'Loading blacklist: {self.blacklist}')

        assert image_pairs
        self.image_pairs = [
            pair for pair in image_pairs
            if (
                str(pair.image1_path) not in self.blacklist
                and str(pair.image2_path) not in self.blacklist
            )
        ]
        self._index = 0

        self.left_pil_image = None
        self.left_photo_image = None

        self._init_ui()
        self._show_current_pair()

    def _init_ui(self):
        self.window = tk.Tk()
        self.window.geometry('1400x800')

        tk.Grid.columnconfigure(self.window, 0, weight=1)
        tk.Grid.columnconfigure(self.window, 1, weight=1)
        tk.Grid.columnconfigure(self.window, 2, weight=1)
        tk.Grid.columnconfigure(self.window, 3, weight=1)

        tk.Grid.rowconfigure(self.window, 0, weight=1)

        self.image_left = tk.Label(self.window)
        self.image_left.grid(row=0, column=0, columnspan=2, sticky='NESW')

        self.image_right = tk.Label(self.window)
        self.image_right.grid(row=0, column=2, columnspan=2, sticky='NESW')

        self.delete_left_button = tk.Button(
            self.window, text='Delete left', command=self.delete_left
        )
        self.delete_left_button.grid(row=1, column=0, columnspan=1, sticky='NESW')

        self.prev_button = tk.Button(self.window, text='Prev', command=self.prev)
        self.prev_button.grid(row=1, column=1, columnspan=1, sticky='NESW')

        self.next_button = tk.Button(self.window, text='Next', command=self.next)
        self.next_button.grid(row=1, column=2, columnspan=1, sticky='NESW')

        self.delete_right_button = tk.Button(
            self.window, text='Delete right', command=self.delete_right
        )
        self.delete_right_button.grid(row=1, column=3, columnspan=1, sticky='NESW')

        self.save_all_button = tk.Button(
            self.window, text='Save duplicates list', command=self.save_deleted
        )
        self.save_all_button.grid(row=2, column=1, columnspan=2, sticky='NESW')

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, i):
        self._index = i
        self.update_title()

    def update_title(self):
        self.window.title(f'{self.index}/{len(self.image_pairs)}')

    def prev(self):
        self.index = max(self.index - 1, 0)
        self._show_current_pair()

    def next(self):
        self.index = min(self.index + 1, len(self.image_pairs) - 1)
        self._show_current_pair()

    def delete_left(self):
        self.blacklist.add(str(self._current_pair.image1_path))
        self.next()

    def delete_right(self):
        self.blacklist.add(str(self._current_pair.image2_path))
        self.next()

    def save_deleted(self):
        self.blacklist.save()

    @property
    def _current_pair(self):
        return self.image_pairs[self.index]

    def _show_current_pair(self):
        pair = self._current_pair

        print(f'Path left: {pair.image1_path}')
        self.left_pil_image = load_img(self.data_root / pair.image1_path)
        self.left_pil_image.thumbnail((800, 600), Image.ANTIALIAS)
        self.left_photo_image = ImageTk.PhotoImage(self.left_pil_image)
        self.image_left.configure(image=self.left_photo_image)

        print(f'Path right: {pair.image2_path}')
        self.right_pil_image = load_img(self.data_root / pair.image2_path)
        self.right_pil_image.thumbnail((800, 600), Image.ANTIALIAS)
        self.right_photo_image = ImageTk.PhotoImage(self.right_pil_image)
        self.image_right.configure(image=self.right_photo_image)

    def run(self):
        self.window.mainloop()

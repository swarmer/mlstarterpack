import argparse
import csv
import os
import pathlib
import tkinter as tk
import tkinter.messagebox as tkmb

from PIL import ImageTk, Image

from mlstarterpack.vision.images import find_images


def read_image_names(csv_paths):
    result = set()

    for csv_path in csv_paths:
        with open(csv_path) as infile:
            reader = csv.reader(infile)

            result.update(name for name, _ in reader)

    return result


def read_annotations(csv_path):
    result = {}

    with open(csv_path) as infile:
        reader = csv.reader(infile)

        for path, score in reader:
            score = float(score)
            result[path] = score

    return result


class ImageSelector:
    def __init__(self, images_root, image_names, output_path, annotations=None):
        self.annotations = annotations or {}
        self.current_image = None
        self.image_names = image_names
        self.images_root = images_root
        self.output_path = output_path
        self.photo_image = None

        assert image_names

        self.window = tk.Tk()
        self.window.geometry('1000x800')

        tk.Grid.columnconfigure(self.window, 0, weight=1)
        tk.Grid.columnconfigure(self.window, 1, weight=1)
        tk.Grid.columnconfigure(self.window, 2, weight=1)

        tk.Grid.rowconfigure(self.window, 0, weight=1)

        self.image_label = tk.Label(self.window)
        self.image_label.grid(row=0, column=0, columnspan=3, sticky='NESW')

        self.scale = tk.Scale(self.window, from_=0, to=100, orient=tk.HORIZONTAL)
        self.scale.grid(row=1, column=0, columnspan=3, sticky='NESW')

        self.prev_button = tk.Button(self.window, text='Prev', command=self.prev)
        self.prev_button.grid(row=2, column=0, columnspan=1, sticky='NESW')

        self.next_button = tk.Button(self.window, text='Next', command=self.next)
        self.next_button.grid(row=2, column=1, columnspan=1, sticky='NESW')

        self.save_all_button = tk.Button(self.window, text='Save all', command=self.save_all)
        self.save_all_button.grid(row=2, column=2, columnspan=1, sticky='NESW')

        self.delete_button = tk.Button(self.window, text='Delete image', command=self.delete)
        self.delete_button.grid(row=3, column=1, columnspan=1, sticky='NESW')

        self.index = 0
        self.set_image(self.image_names[0])

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, i):
        self._index = i
        self.update_title()

    def update_title(self):
        self.window.title(f'{self.index}/{len(self.image_names)}')

    def prev(self):
        self.save()

        self.index = max(self.index - 1, 0)
        self.set_image(self.image_names[self.index])

    def next(self):
        self.save()

        self.index = min(self.index + 1, len(self.image_names) - 1)
        self.set_image(self.image_names[self.index])

    def save(self):
        self.annotations[str(self.image_names[self.index])] = self.scale.get() / 100

    def delete(self):
        if not tkmb.askokcancel('Delete an image', 'Are you sure you want to delete this image?'):
            return

        image_path = str(self.images_root / self.image_names[self.index])
        if image_path in self.annotations:
            del self.annotations[image_path]
        del self.image_names[self.index]
        os.remove(image_path)

        if self.index == len(self.image_names):
            self.index -= 1

        self.update_title()
        self.set_image(self.image_names[self.index])

    def save_all(self):
        with open(self.output_path, 'w') as outfile:
            writer = csv.writer(outfile)

            for path, value in self.annotations.items():
                writer.writerow([path, str(value)])

    def set_image(self, image_name):
        self.current_image = Image.open(str(self.images_root / image_name))
        self.current_image.thumbnail((800, 600), Image.ANTIALIAS)
        self.photo_image = ImageTk.PhotoImage(self.current_image)

        self.image_label.configure(image=self.photo_image)
        self.scale.set(self.annotations.get(str(image_name), 0) * 100)

    def run(self):
        self.window.mainloop()


class UniverseSet:
    def __contains__(self, item):
        return True

    def __iter__(self):
        return iter([])


def parse_arguments():
    parser = argparse.ArgumentParser(description='Annotate images with a single quality number')
    parser.add_argument(
        '--images-folder', required=True, type=pathlib.Path,
        help='Path to the folder with images to be annotated',
    )
    parser.add_argument(
        '--images-root', required=True, type=pathlib.Path,
        help='Anchor image path',
    )
    parser.add_argument(
        '--annotation-path', required=True, type=pathlib.Path,
        help='Path to the resulting annotation CSV file',
    )
    parser.add_argument(
        '--exclude', required=False, action='append', type=pathlib.Path,
        help='Exclude images from specified annotation file',
    )
    parser.add_argument(
        '--refine', required=False, action='store_true', default=False,
        help='Only show images from an already existing <annotation-path> file',
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.exclude:
        exclude_names = read_image_names(args.exclude)
    else:
        exclude_names = set()

    if args.refine:
        annotations = read_annotations(args.annotation_path)
        include_names = frozenset(annotations.keys())
    else:
        annotations = {}
        include_names = UniverseSet()

    image_names = []
    for image_path in find_images(args.images_folder):
        image_name = image_path.relative_to(args.images_root.resolve())

        if image_name in exclude_names or image_name not in include_names:
            continue

        image_names.append(image_name)
    print(f'{len(image_names)} images to classify')

    images_not_found = frozenset(include_names) - frozenset(image_names)
    if images_not_found:
        print(f'Warning: images cannot be found: {list(images_not_found)}')

    selector = ImageSelector(args.images_root, image_names, args.annotation_path, annotations)
    selector.run()


if __name__ == '__main__':
    main()

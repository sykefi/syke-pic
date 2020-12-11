"""Module for the PredictionViewer"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from ipywidgets import Box, Button, Dropdown, Image, Label, Layout, Text, VBox

from .classification import read_predictions
from sykepic.predict import ifcb


class PredictionViewer():
    """Graphical tool for inspecting, evaluating and labeling predictions.

    This tool is meant to be used in a Jupyter Notebook, since
    it uses IPython and ipywidgets in the background.

    Parameters
    ----------
    predictions : str, Path, list
        Path to csv-file with predictions. You can provide
        more than one prediction file when doing class inspection.
        Only one file is allowed at a time for evaluation and labeling.
    raw_dir : str, Path
        Root directory of raw IFCB data
    work_dir : str, Path
        Directory used to extract sample images and store progress files.
        If left empty, a default directory is created
        in ~/PredictionViewer. Work directory is removed if it
        remains empty after program exits.
    label : bool
        Use viewer to perform active labeling
    evaluate : bool
        Use viewer to evaluate predictions
    thresholds : float, str, Path
        Single value or file with classification thresholds for each class
    empty : str
        Name to use for unclassifiable images
    keep_images : bool
        Don't remove extracted images, which is done by default

    Methods
    -------
    start(per_page=12, sort_by_confidence=True, sort_by_class=False,
          ascending=False, class_overview=False, prediction_filter=None,
          inverse_prediction_filter=True, start_page=1):
    """

    def __init__(self, predictions, raw_dir, work_dir='PredictionViewer',
                 label=False, evaluate=False, thresholds=0.0,
                 empty='unclassifiable', keep_images=False):

        if label and evaluate:
            raise ValueError('Choose either label or evaluate, not both.')

        self.work_dir = Path(work_dir)
        # This is not the best way to handle img sub dirs
        if isinstance(predictions, list):
            assert not (label or evaluate), \
                'Labeling and evaluation is allowed one sample a time'
            self.img_dir = {}
            for csv in predictions:
                sample = Path(csv).with_suffix('').name
                self.img_dir[sample] = self.work_dir/'images'/sample
        else:
            self.sample = Path(predictions).with_suffix('').name
            self.img_dir = self.work_dir/'images'/self.sample

        self.df = read_predictions(predictions, thresholds)

        self.raw_dir = raw_dir
        self.label = label
        self.evaluate = evaluate
        self.select = self.label or self.evaluate
        self.keep_images = keep_images
        self.labeled = {}
        self.moved = []
        self.empty = empty

        if self.select:
            # Update progress or start new
            if self.evaluate:
                self.sel_dir = self.work_dir/'evaluate'
            else:
                self.sel_dir = self.work_dir/'label'
            self.sel_log = self.sel_dir/f'{self.sample}.select.csv'
            self.moved_log = self.sel_dir/f'{self.sample}.copied.csv'
            if self.label and self.moved_log.is_file():
                with open(self.moved_log) as fh:
                    for i, line in enumerate(fh, start=1):
                        roi, label = line.strip().split(',')
                        self.moved.append(int(roi))
                print(f'[INFO] Skipped {i} previously labeled images in '
                      f"'{self.moved_log}'")
            if self.sel_log.is_file():
                print('[INFO] Using previous selections for this '
                      f'sample from:\n\t{self.sel_log}')
                print('[INFO] Remove it manually to start from scratch')
                with open(self.sel_log) as fh:
                    for line in fh:
                        roi, name = line.strip().split(',')
                        self.labeled[int(roi)] = name
            else:
                print(
                    f"[INFO] Creating a new progress file in '{self.sel_log}'")
                Path.mkdir(self.sel_dir, parents=True, exist_ok=True)
            if self.label:
                self.extra_labels = ['New_Class']
                # Add any novel class labels found in work_dir
                for p in self.sel_dir.iterdir():
                    if p.is_dir():
                        cond_1 = p.name not in self.extra_labels
                        cond_2 = p.name not in self.df.columns[2:]
                        cond_3 = p.name != self.empty
                        if all((cond_1, cond_2, cond_3)):
                            self.extra_labels.append(p.name)

        self.item_layout = Layout(
            display='flex',
            flex_flow='column',
            align_items='center'
        )
        self.item_container_layout = Layout(
            dispay='flex',
            flex_flow='row wrap',
            align_items='baseline'
        )
        self.button_container_layout = Layout(
            display='flex ',
            flex_flow='row',
            justify_content='space-around',
            align_items='flex-end',
            margin='30px 0'
        )
        self.main_container_layout = Layout(
            display='flex',
            flex_flow='column',
            justify_content='space-between'
        )

    def start(self, per_page=12, sort_by_confidence=True, sort_by_class=False,
              ascending=False, class_overview=False, prediction_filter=None,
              inverse_prediction_filter=True, start_page=1):
        """Start the program

        Parameters
        ----------
        per_page : int
            Number of images to display per page
        sort_by_confidence : bool
            Sort images by prediction confidence (default)
        sort_by_class : bool
            Sort images by classification
        ascending : bool
            Sort in an ascending order
        class_overview : bool
            Display each class on a separate page. If the number of
            images per class is bigger than `per_page`,
            a representative sample of them will be selected. This is
            useful for inspection, but should not be used in evaluation.
        prediction_filter : str, list
            Filter images by prediction (class name)
        inverse_prediction_filter : bool
            Use prediction_filter to exclude classes
        start_page : int
            Select at which page to start
        """

        # Extract images only if they don't already exist
        # Multiple samples
        if isinstance(self.img_dir, dict):
            for sample, img_dir in self.img_dir.items():
                if not img_dir.is_dir() or len(list(img_dir.iterdir())) == 0:
                    print(f'[INFO] Extracting images for {sample}')
                    ifcb.extract_sample_images(
                        sample, self.raw_dir, img_dir, True)
        # One sample
        else:
            if (not self.img_dir.is_dir() or
                    len(list(self.img_dir.iterdir())) == 0):
                ifcb.extract_sample_images(
                    self.sample, self.raw_dir, self.img_dir, True)

        # Filter predictions by class
        if prediction_filter:
            if isinstance(prediction_filter, str):
                prediction_filter = [prediction_filter]
            for name in prediction_filter:
                assert name in self.df.columns[2:], f"Unknown class '{name}'"
            if inverse_prediction_filter:
                df = self.df[~self.df['prediction'].isin(prediction_filter)]
            else:
                df = self.df[self.df['prediction'].isin(prediction_filter)]
            if len(df) <= 0:
                print('[INFO] No predictions to show')
                return
        else:
            df = self.df

        if self.label:
            # Don't show previously labeled and moved images
            df = df.drop(self.moved)

        # Divide predictions to pages
        self.pages = []
        if sort_by_class or class_overview:
            # Group data frame by prediction classes
            for name, group in df.sort_values('prediction').groupby('prediction'):
                if len(group) < 1:
                    continue
                group_indeces = group.sort_values(
                    name, ascending=ascending).index
                num_preds = len(group_indeces)
                if class_overview:
                    # Choose random subset of indeces from inside each
                    # prediction group, across different confidence values
                    # Show at max, 'per_page' number of predictions per class
                    num_preds_to_display = min(num_preds, per_page)
                    subset = np.linspace(
                        0, num_preds-1, num=num_preds_to_display, dtype=int)
                    # Create one page for each prediction group
                    self.pages.append(group_indeces[subset])
                else:
                    # Create as many pages as it takes for this class
                    self.pages.extend([group_indeces[i:i+per_page]
                                       for i in range(0, num_preds, per_page)])
        else:
            # Sort all predictions by confidence
            if sort_by_confidence:
                df['confidence'] = df.apply(
                    lambda row: row[row['prediction']], axis=1)
                df.sort_values('confidence', ascending=ascending, inplace=True)
                df.drop('confidence', axis=1, inplace=True)
            # Sort all predictions by roi number (index)
            else:
                df.sort_index(ascending=ascending, inplace=True)
            self.unlabeled = df.index.tolist()
            # Divide indeces to equal sized pages
            for i in range(0, len(self.unlabeled), per_page):
                self.pages.append(self.unlabeled[i:i+per_page])

        self.current_page = start_page - 1
        self._show_current_page()

    def _show_current_page(self, *args):
        clear_output()
        if not self.pages:
            print('[INFO] No predictions to show')
            return
        next_disabled = True if self.current_page == len(
            self.pages)-1 else False
        back_disabled = True if self.current_page == 0 else False
        unlabeled = self.pages[self.current_page]
        items = []
        for roi in unlabeled:
            item = self._new_item(roi)
            items.append(item)
        next_btn = Button(description='>', button_style='info',
                          disabled=next_disabled, tooltip='Show next page')
        next_btn.on_click(self._next_button_handler)
        back_btn = Button(description='<', button_style='info',
                          disabled=back_disabled, tooltip='Show previous page')
        back_btn.on_click(self._back_button_handler)
        end_tooltip = 'End labeling'
        if not self.keep_images:
            end_tooltip += ' and remove extracted images'
        end_btn = Button(description='End', button_style='warning',
                         tooltip=end_tooltip)
        end_btn.on_click(self._end_button_handler)
        item_container = Box(children=items, layout=self.item_container_layout)
        button_container = Box(children=[end_btn, back_btn, next_btn],
                               layout=self.button_container_layout)
        main_container = Box(children=[item_container, button_container],
                             layout=self.main_container_layout)
        display(main_container)
        print(f'Page {self.current_page+1} / {len(self.pages)}')
        if self.select:
            self._log_selections()

    def _new_item(self, roi):
        row = self.df.loc[roi]
        prediction = row['prediction'] if row['classified'] else self.empty
        confidence = row[row['prediction']]
        probs = row[2:].sort_values(ascending=False)

        if isinstance(self.img_dir, dict):
            # Extract multi-index
            sample, roi_num = roi
            try:
                with open(self.img_dir[sample]/f'{roi_num}.png', 'rb') as fh:
                    img = Image(value=fh.read(), format='png')
            except FileNotFoundError:
                print('[ERROR] Images have likely been moved due to labeling')
                print(f'[INFO] Try deleting {self.work_dir}/images')
                return
            # Label to display below each image
            label = Label(f'{confidence:.5f}')
        else:
            try:
                with open(self.img_dir/f'{roi}.png', 'rb') as fh:
                    img = Image(value=fh.read(), format='png')
            except FileNotFoundError:
                print('[ERROR] Images have likely been moved due to labeling.')
                print(f'[INFO] Try deleting {self.work_dir}/images')
                return
            label = Label(f'{roi} - {confidence:.5f}')

        if self.select:
            if roi in self.labeled:
                # This image has been seen before
                previous = self.labeled[roi]
                if previous != prediction:
                    # Previous selection was a correction
                    # Add check mark to distinguish this
                    label.value = '\N{White Heavy Check Mark} ' + label.value
                    prediction = previous
            else:
                self.labeled[roi] = prediction
        # Set dropdown selection to empty initially
        options = ['']
        predicted_option = ''
        for name, prob in probs.items():
            option = f'{prob:.5f} - {name}'
            options.append(option)
            if name == prediction:
                # Set dropdown selection to predicted class
                predicted_option = option

        if self.label:
            options.insert(1, *self.extra_labels)
        dropdown = Dropdown(options=options, value=predicted_option)
        if self.select:
            # Add listener to dropdown menu
            dropdown.observe(self._dropdown_handler(roi), names='value')
        item = Box(children=[img, label, dropdown], layout=self.item_layout)
        return item

    def _cleanup(self):
        if self.keep_images:
            return
        if isinstance(self.img_dir, dict):
            for img_dir in self.img_dir.values():
                shutil.rmtree(img_dir)
                print(f"[INFO] Images removed from '{img_dir}'")
        else:
            shutil.rmtree(self.img_dir)
            print(f"[INFO] Images removed from '{self.img_dir}'")
        try:
            self.work_dir.rmdir()
            print(f"[INFO] Removed empty directory '{self.work_dir}'")
        except OSError:
            pass

    def _next_button_handler(self, button):
        self.current_page += 1
        self._show_current_page()

    def _back_button_handler(self, button):
        self.current_page -= 1
        self._show_current_page()

    def _end_button_handler(self, button):
        clear_output()
        if self.select:
            self._log_selections()
        if not self.label or not self.labeled:
            self._cleanup()
            return
        if self.label:
            num_labeled = len([name for name in self.labeled.values()
                               if name != self.empty])
            print(f'[INFO] Total labeled images: {num_labeled}')
            dest = Text(value=str(self.sel_dir),
                        description='Copy to:',
                        tooltip='Each label gets a sub-directory automatically')
            move_btn = Button(description='Accept', button_style='success',
                              tooltip='Copy labeled images and exit')
            move_btn.dest = dest
            move_btn.on_click(self._move_button_handler)
            back_btn = Button(description='Back', button_style='info',
                              tooltip='Back to previous page')
            back_btn.on_click(self._show_current_page)
            quit_btn = Button(description='Quit', button_style='danger',
                              tooltip='Exit without saving labels')
            quit_btn.on_click(self._quit_button_handler)
            buttons = Box(children=[quit_btn, back_btn, move_btn],
                          layout=Layout(margin='30px 0 10px 0'))
            display(VBox([dest, buttons]))

    def _move_button_handler(self, button):
        clear_output()
        dest_dir = Path(button.dest.value)
        with open(self.moved_log, 'a') as fh:
            for roi, label in self.labeled.items():
                if not label or label == self.empty:
                    # Don't copy unlabeled images
                    continue
                src = self.img_dir/f'{roi}.png'
                dst = dest_dir/label/f'{self.sample}_{roi}.png'
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
                fh.write(f'{roi},{label}\n')
                print(f'[INFO] Copied {roi} to {dst}')
                self.moved.append(roi)
        self.labeled = {}
        self.sel_log.unlink()
        self._cleanup()

    def _quit_button_handler(self, button):
        clear_output()
        self._cleanup()

    def _dropdown_handler(self, roi):
        def handler(change):
            self.labeled[roi] = change.new.split(' - ')[-1]
        return handler

    def _log_selections(self):
        data = ''
        for roi, selection in self.labeled.items():
            if not selection:
                # Set empty selection to a string
                selection = self.empty
            data += f"{roi},{selection}\n"
        with open(self.sel_log, 'w') as fh:
            fh.write(data)

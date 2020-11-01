"""Module for the PredictionViewer"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from ipywidgets import Box, Button, Dropdown, Image, Label, Layout, Text, VBox

from .dataframe import insert_pred_conf
from sykepic.predict import ifcb
from sykepic.analysis.dataframe import read_thresholds


class PredictionViewer():
    """Graphical tool for prediction inspection and active labeling.

    This tool is meant to be used in a Jupyter Notebook, since
    it uses IPython and ipywidgets in the background.

    Parameters
    ----------
    prediction_csv : str, Path, list
        Path to csv-file with predictions. You can provide
        more than one prediction file when doing class inspection.
        Only one file is allowed at a time for active labeling.
    raw_dir : str, Path
        Root directory of raw IFCB data
    work_dir : str, Path
        Directory used to extract sample images and store labeled
        images. If left empty, a default directory is created
        in ~/PredictionViewer. Work directory is removed if it
        remains empty after program exits.
    label : bool
        Use viewer to perform active labeling
    evaluate : bool
        Use viewer to evaluate predictions
    thresholds : str, Path
        File with classification thresholds for each class
    keep_images : bool
        Don't remove extracted images, which is done by default

    Methods
    -------
    start(per_page=9, sort_by_confidence=True, ascending=False,
          prediction_filter=None, class_inspect=False)
    """

    def __init__(self, prediction_csv, raw_dir, work_dir='PredictionViewer',
                 label=False, evaluate=False, thresholds=None, keep_images=False):
        self.work_dir = Path(work_dir)
        # Read csv to pandas DataFrame
        if isinstance(prediction_csv, list):
            assert not (label or evaluate), \
                'Labeling and evaluation is allowed one sample a time'
            df_list = []
            # self.sample = []
            self.img_dir = {}
            for csv in prediction_csv:
                sample = Path(csv).with_suffix('').name
                df = pd.read_csv(csv)  # .drop('roi', 1)
                df.insert(0, 'sample', sample)
                # Create multi-index from sample name and roi number
                df.set_index(['sample', 'roi'], inplace=True)
                df_list.append(df)
                self.img_dir[sample] = self.work_dir/'images'/sample
            self.df = pd.concat(df_list)
        else:
            self.df = pd.read_csv(prediction_csv, index_col=0)
            self.sample = Path(prediction_csv).with_suffix('').name
            self.img_dir = self.work_dir/'images'/self.sample

        # Insert 'prediction' and 'confidence' columns to dataframe
        insert_pred_conf(self.df)

        self.raw_dir = raw_dir
        self.label = label
        self.evaluate = evaluate
        self.thresholds = {}
        self.keep_images = keep_images
        self.labeled = {}
        self.moved = []
        self.eval_dict = {}

        if self.label:
            # Update progress or start new
            log_dir = self.work_dir/'.labeled'
            self.log = log_dir/f'{self.sample}_labeled.csv'
            if self.log.is_file():
                with open(self.log) as fh:
                    for i, line in enumerate(fh, start=1):
                        roi, label = line.strip().split(',')
                        self.moved.append(int(roi))
                print(
                    f"[INFO] Skipped {i} previously labeled images in '{self.log}'")
            else:
                print(f"[INFO] Creating a new progress file in '{self.log}'")
                Path.mkdir(log_dir, parents=True, exist_ok=True)
            self.extra_labels = ['', 'New_Class', ]
            # Add any novel class labels found in work_dir
            for p in self.work_dir.iterdir():
                if p.is_dir():
                    cond_1 = not p.name.startswith('.')
                    cond_2 = not p.name.endswith('images')
                    cond_3 = p.name not in self.extra_labels
                    cond_4 = p.name not in self.df.columns[2:]
                    if all((cond_1, cond_2, cond_3, cond_4)):
                        self.extra_labels.append(p.name)

        elif self.evaluate:
            self.eval_log = self.work_dir/f'{self.sample}.eval.log'
            self.df = read_thresholds(self.df, thresholds, filter=False)

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
              ascending=False, prediction_filter=None, class_overview=False):
        """Start the program

        Parameters
        ----------
        per_page : int
            Number of images to display per page
        class_inspect : bool
            Display each prediction class on a separate page.
            This is useful for inspecting class confidence thresholds.
            If the number of predictions per class is bigger than
            `per_page`, a representative sample of them will be selected.
        sort_by_confidence : bool
            Sort images by prediction confidence (default).
            If set to False, predictions are ordered by ROI.
        ascending : bool
            Sort in an ascending order.
        prediction_filter : str, list
            Filter images by prediction (class name)
        """

        # assert not (class_inspect and self.label), \
        #     'Class inspection is not available when labeling'

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
                group_indeces = group.sort_values(
                    'confidence', ascending=ascending).index
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
                df.sort_values('confidence', ascending=ascending, inplace=True)
            # Sort all predictions by roi number (index)
            else:
                df.sort_index(ascending=ascending, inplace=True)
            self.unlabeled = df.index.tolist()
            # Divide indeces to equal sized pages
            for i in range(0, len(self.unlabeled), per_page):
                self.pages.append(self.unlabeled[i:i+per_page])

        self.current_page = 0
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
        if self.evaluate:
            self._log_evaluations()

    def _new_item(self, roi):
        row = self.df.loc[roi]
        confidence = row['confidence']
        if self.evaluate:
            # thresholds column was added to df, so ignore it
            probs = row[2:-1].sort_values(ascending=False)
        else:
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
        if self.label:
            options = self.extra_labels.copy()
        elif self.evaluate:
            options = ['']
        else:
            options = []
        options.extend(
            [f'{prob:.5f} - {name}' for name, prob in probs.items()])
        dropdown = Dropdown(
            options=options, value=self.labeled.get(roi, options[0]))
        # DISABLED FEATURE
        # Add a hidden textbox for entering new class names.
        # This textbox and dropdown are linked, and the textbox will become
        # visible once 'New Class' is selected from dropdown.
        # textbox = Text(placeholder='Enter new class name here')
        # textbox.layout.visibility = 'hidden'
        # dropdown.textbox = textbox
        if self.label:
            # Add listener to dropdown menu
            dropdown.observe(self._dropdown_handler(roi), names='value')
        # item_children = [img, label, dropdown]
        if self.evaluate:
            # Just select class from dropdown if above threshold, otherwise empty selection.
            # When user changes selection, it will be handled accordingly
            # i.e.
            # this  -> empty : FP
            # this  -> other : FP + FN (other)
            # empty -> this  : FN
            # empty -> other : FN (other)

            pred_class = 'unclassifiable'
            if confidence >= row['threshold']:
                pred_class = probs.index[0]
                dropdown.value = options[1]
            self.eval_dict.setdefault(roi, (pred_class, pred_class))
            dropdown.observe(
                self._dropdown_handler_eval(roi, pred_class), names='value')
            if roi in self.eval_dict:
                # ROI has already been processed (on a previous page)
                pred_class, selected_class = self.eval_dict[roi]
                # Check if user had changed ROI's class
                if pred_class != selected_class:
                    # Add check mark to label
                    label.value = '\N{White Heavy Check Mark} ' + label.value
                    # Set previous dropdown selection
                    if selected_class == 'unclassifiable':
                        dropdown.value = ''
                    else:
                        for option in options:
                            if selected_class in option:
                                dropdown.value = option
                                break

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
        if not self.label or not self.labeled:
            if self.evaluate:
                self._log_evaluations()
            print('[INFO] Program exited with no images labeled')
            self._cleanup()
            return
        print(f'[INFO] Total labeled images: {len(self.labeled)}')
        dest = Text(value=str(self.work_dir),
                    description='Move to:',
                    tooltip='Each label gets a sub-directory automatically')
        move_btn = Button(description='Accept', button_style='success',
                          tooltip='Move labeled images and exit')
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
        with open(self.log, 'a') as fh:
            for roi, label in self.labeled.items():
                label = label.split(' - ')[-1]
                src = self.img_dir/f'{roi}.png'
                dst = dest_dir/label/f'{self.sample}_{roi}.png'
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src, dst)
                fh.write(f'{roi},{label}\n')
                print(f'[INFO] Moved {roi} to {dst}')
                # In case user wants to start again
                self.moved.append(roi)
        self.labeled = {}
        self._cleanup()

    def _quit_button_handler(self, button):
        clear_output()
        print('[INFO] Program exited with unsaved labels')
        self._cleanup()

    def _dropdown_handler(self, roi):
        def handler(change):
            # if change.new == 'New Class':
            #     textbox.layout.visibility = 'visible'
            if change.new == '':
                del self.labeled[roi]
            else:
                # textbox.layout.visibility = 'hidden'
                self.labeled[roi] = change.new

        return handler

    def _dropdown_handler_eval(self, roi, pred_class):
        def handler(change):
            selected_class = change.new.split(' - ')[-1]
            self.eval_dict[roi] = (
                pred_class if pred_class else 'unclassifiable',
                selected_class if selected_class else 'unclassifiable')
        return handler

    def _log_evaluations(self):
        data = 'roi,prediction,actual\n'
        for roi, values in self.eval_dict.items():
            data += f"{roi},{values[0]},{values[1]}\n"
        with open(self.eval_log, 'w') as fh:
            fh.write(data)

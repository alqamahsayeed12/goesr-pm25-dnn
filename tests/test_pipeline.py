import contextlib
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from src import pipeline
from src.config import PATHS
from src.download_scalars import download_from_gdrive
from src.preprocess import szafunc


class ConstantModel:
    def predict(self, values, **kwargs):
        return np.full((len(values), 1), 10., dtype='float32')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = {**PATHS, **{key: self.root / key for key in
                      ['scalar_folder', 'hrrr_folder', 'output_nc', 'plots', 'goes_folder']}}
        for key in ['scalar_folder', 'hrrr_folder', 'output_nc', 'plots', 'goes_folder']:
            self.paths[key].mkdir()
        original_scalars = PATHS['scalar_folder']
        overrides = patch.dict(pipeline.PATHS, self.paths)
        overrides.start()
        self.addCleanup(overrides.stop)
        y, x = np.indices((3, 3))
        self.base = pd.DataFrame({'time': 1, 'projection_y_coordinate': y.ravel(),
                                  'projection_x_coordinate': x.ravel()})
        self.base.to_csv(self.paths['scalar_folder'] / 'base_df.csv')
        self.collocated = pd.DataFrame({'goes_i': y.ravel(), 'goes_j': x.ravel(),
                         'hrrr_i': y.ravel(), 'hrrr_j': x.ravel(),
                         'lon_com': -75., 'lat_com': 35., 'hrrr_lat': 35., 'hrrr_lon': -75.})
        self.features = ['Lat', 'Lon', 'SED', 'SZA', 'Wind Speed', 'RH', 'TMP',
                         'smoke_dust_mask_ge', 'aod_avg', 'pm_avg']
        self.dnn = [f'DNN_{i:02}' for i in range(14)]
        self.features2 = self.features + self.dnn
        ranges = pd.read_csv(original_scalars / 'max_min.csv', index_col=0)
        self.mx, self.mn = ranges[['mx']].T, ranges[['mn']].T
        self.mx[self.dnn], self.mn[self.dnn] = 1000., 0.
        self.mx['aod_avg'], self.mn['aod_avg'] = 4.920678139, 0.
        self.vars = ['MAXUW_1hr_max_fcst', 'MAXVW_1hr_max_fcst', 'RH', 'TMP']
        self.metadata = pd.DataFrame({'ZARR_LEVEL_NAME': ['surface']*4,
                                      'Parameter Short Name': self.vars})
        self.file = 'pm25_gwr_aod_exp50_2021011313.nc'
        self.output = self.paths['output_nc'] / 'pm25_gwr_aod_exp50_2021011313_dnn_v1_4.nc'
        self.cache = self.paths['hrrr_folder'] / '20210113_12z_df.csv'

    def merged_goes(self, _):
        names = ['pm25sat_com', 'abi_aod', 'smoke_dust_mask']
        return xr.Dataset({k: (('ydim', 'xdim'), np.ones((3, 3), dtype='float32')) for k in names},
                     coords={'lon_com': (('ydim', 'xdim'), np.full((3, 3), -75.)),
                             'lat_com': (('ydim', 'xdim'), np.full((3, 3), 35.))})

    def hrrr_source(self, url, **kwargs):
        variable = url.rstrip('/').split('/')[-2]
        return xr.Dataset({variable: (('time', 'projection_y_coordinate', 'projection_x_coordinate'),
                          np.ones((1, 3, 3), dtype='float32'))},
                          coords={'time': [1], 'projection_y_coordinate': range(3),
                                  'projection_x_coordinate': range(3)})

    def run_pipeline(self, plot=False):
        pipeline.run_pipeline(self.file, self.collocated, self.metadata, range(4),
               self.features, self.features2, self.mx, self.mn, self.dnn, plot=plot)

    def test_first_run_predicts_and_cached_run_matches(self):
        inputs = []
        class RecordingModel(ConstantModel):
            def predict(inner, values, **kwargs):
                inputs.append(values.copy())
                return super().predict(values, **kwargs)
        with patch.object(pipeline, 'merge_GOES', side_effect=self.merged_goes), \
             patch.object(pipeline.xr, 'open_dataset', side_effect=self.hrrr_source) as source, \
             patch.object(pipeline.keras.models, 'load_model', return_value=RecordingModel()):
            self.run_pipeline()
            self.assertEqual(source.call_count, 4)
            self.assertTrue(self.cache.is_file())
            self.assertTrue(self.output.is_file())
            self.assertEqual(len(inputs), 15)
            expected = (szafunc(pd.Timestamp('2021-01-13 13:00'), -75., 35.) - self.mn.SZA.iloc[0]) / (self.mx.SZA.iloc[0] - self.mn.SZA.iloc[0])
            np.testing.assert_allclose(inputs[0]['SZA'], expected)
        with xr.open_dataset(self.output) as ds:
            first = ds.pm25gwr_dnn_com_ensemble.values.copy()
            self.assertTrue(np.isfinite(first).all())
        self.output.unlink()
        with patch.object(pipeline, 'merge_GOES', side_effect=self.merged_goes), \
             patch.object(pipeline.keras.models, 'load_model', return_value=ConstantModel()):
            self.run_pipeline()
        with xr.open_dataset(self.output) as ds:
            np.testing.assert_array_equal(first, ds.pm25gwr_dnn_com_ensemble.values)

    def test_processing_error_propagates_before_plotting(self):
        for plot in [False, True]:
            with self.subTest(plot=plot), \
                 patch.object(pipeline, 'merge_GOES', side_effect=ValueError('bad GOES')), \
                 patch.object(pipeline.xr, 'open_dataset') as open_data:
                with self.assertRaisesRegex(RuntimeError, 'Failed to process GOES') as error:
                    self.run_pipeline(plot=plot)
                self.assertIsInstance(error.exception.__cause__, ValueError)
                open_data.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_hrrr_error_does_not_save_cache(self):
        with patch.object(pipeline, 'merge_GOES', side_effect=self.merged_goes), \
             patch.object(pipeline.xr, 'open_dataset', side_effect=OSError('unavailable')):
            with self.assertRaises(RuntimeError) as error:
                self.run_pipeline(plot=True)
            self.assertIn('Failed to retrieve HRRR variable', str(error.exception.__cause__))
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.output.exists())

    def test_help_does_not_download(self):
        with patch.object(sys, 'argv', ['main.py', '--help']), \
             patch('src.download_scalars.download_from_gdrive') as download, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                runpy.run_path(str(Path(__file__).resolve().parents[1] / 'main.py'), run_name='__main__')
            self.assertEqual(error.exception.code, 0)
            download.assert_not_called()

    def test_failed_cache_write_is_not_reused(self):
        def fail_write(frame, path, **kwargs):
            Path(path).write_text('incomplete')
            raise OSError('disk write failed')
        with patch.object(pipeline, 'merge_GOES', side_effect=self.merged_goes), \
             patch.object(pipeline.xr, 'open_dataset', side_effect=self.hrrr_source), \
             patch.object(pd.DataFrame, 'to_csv', fail_write):
            with self.assertRaises(RuntimeError):
                self.run_pipeline()
        self.assertFalse(self.cache.exists())
        self.assertFalse(Path(str(self.cache) + '.part').exists())

    def test_failed_output_write_is_not_reused_or_plotted(self):
        def fail_write(dataset, path, **kwargs):
            Path(path).write_text('incomplete')
            raise OSError('disk write failed')
        with patch.object(pipeline, 'merge_GOES', side_effect=self.merged_goes), \
             patch.object(pipeline.xr, 'open_dataset', side_effect=self.hrrr_source) as source, \
             patch.object(pipeline.keras.models, 'load_model', return_value=ConstantModel()), \
             patch.object(xr.Dataset, 'to_netcdf', fail_write):
            with self.assertRaises(RuntimeError):
                self.run_pipeline(plot=True)
            self.assertEqual(source.call_count, 4)
        self.assertFalse(self.output.exists())
        self.assertFalse(Path(str(self.output) + '.part').exists())

    def test_no_goes_files_fails(self):
        with patch.object(sys, 'argv', ['main.py', '--start', '20210113', '--end', '20210113']), \
             patch('src.download_scalars.download_from_gdrive'), \
             patch('src.download.download_goes'), patch('glob.glob', return_value=[]):
            with self.assertRaisesRegex(FileNotFoundError, 'No GOES input files'):
                runpy.run_path(str(Path(__file__).resolve().parents[1] / 'main.py'), run_name='__main__')

    def test_scalar_download_failure_propagates(self):
        with patch('src.download_scalars.gdown.download', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Failed to download required input'):
                download_from_gdrive('not-a-real-id', filename='missing-test-input.csv')


if __name__ == '__main__':
    unittest.main()

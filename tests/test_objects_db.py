import unittest

from multimodal.db.models.objects import Frame


class TestParser(unittest.TestCase):

    one_obj = '0_01_o01_1372086405.972948 | 10 6 301 336'
    three_obj = ('0_01_o01_1372086405.972948 | 204 186 200 147 | '
                 '204 186 200 147 | 204 186 200 147')

    def test_one_object_filename(self):
        frame = Frame.from_line(self.one_obj)
        self.assertEqual(frame.filename, '0_01_o01_1372086405.972948')

    def test_one_object_objects(self):
        frame = Frame.from_line(self.one_obj)
        self.assertEqual(frame.views, [(10, 6, 301, 336)])

    def test_three_object_filename(self):
        frame = Frame.from_line(self.three_obj)
        self.assertEqual(frame.filename, '0_01_o01_1372086405.972948')

    def test_label(self):
        frame = Frame.from_line(self.three_obj)
        self.assertEqual(frame.label, 1)

    def test_time(self):
        frame = Frame.from_line(self.three_obj)
        self.assertEqual(frame.timestamp, 1372086405.972948)

    def test_three_object_objects(self):
        frame = Frame.from_line(self.three_obj)
        self.assertEqual(len(frame.views), 3)
        self.assertEqual(frame.views[1], (204, 186, 200, 147))

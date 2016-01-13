import os
import xml.dom.minidom as dom

from .sound import Record, DataBase


def check_year(year):
    if year not in [1, 2]:
        raise(ValueError, "Wrong year version: %d (should be 1 or 2)" % year)


def _filter_XML(path):
    return (os.path.isfile(path)
            and os.path.splitext(path)[-1].lower() == '.xml')


def _get_XML_files_in(dirname):
    """Returns list of xml files in given directory
    (detection is extension based).
    """
    file_paths = [os.path.join(dirname, f) for f in os.listdir(dirname)]
    return filter(_filter_XML, file_paths)


def _get_tag_node_name(year):
    return 'vistag' + ('' if year == 1 else 's')


class AcornsDB(DataBase):

    XML_DIR = 'XML'
    WAV_DIR = 'WAV'
    WAV_EXT = '.wav'

    def __init__(self):
        DataBase.__init__(self)

    def from_ACORNS_root(self, root, year=1):
        """
        @param version: Acorns version, 1 for year 1, 2 for year 2 (default)
        """
        check_year(year)
        root = os.path.abspath(root)
        tag_node_name = _get_tag_node_name(year)
        self.root = root
        for s in self.get_speakers(year):
            spk_id = self.add_speaker(s)
            # Parse files and populate records
            spk_root = os.path.join(self.root, self.XML_DIR, s)
            for xml_file in _get_XML_files_in(spk_root):
                rec = self._parse_record(spk_id, s, xml_file, tag_node_name)
                self.add_record(rec)
        self.sort()

    def _parse_record(self, speaker_id, speaker, xml_file, tag_node_name):
        parsed = dom.parse(os.path.join(self.root, self.XML_DIR, speaker,
                                        xml_file))
        # Ignore other than first utterance
        utt = parsed.getElementsByTagName('utterance')[0]
        style = utt.getElementsByTagName('style')[0].getAttribute('value')
        audio = utt.getElementsByTagName(
            'audio-file'
            )[0].getAttribute('value') + self.WAV_EXT
        tag_names = utt.getElementsByTagName(
            tag_node_name
            )[0].childNodes[0].data
        tags = [self.get_tag_add(tn)
                for tn in tag_names.split()]
        trans = utt.getElementsByTagName(
            'trans'
            )[0].childNodes[2].data.strip()
        return Record(self, speaker_id, audio, tags, trans, style)

    @classmethod
    def n_speakers(cls, year):
        return 4 if year == 1 else 10

    @classmethod
    def get_speakers(cls, year):
        n_speakers = cls.n_speakers(year)
        return ['Speaker-%.2d' % i for i in range(1, 1 + n_speakers)]

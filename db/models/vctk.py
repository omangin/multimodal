import os

from .sound import Record, DataBase


STYLE = 'ADS'


def _parse_speaker_info(line):
    l = line.strip().split()
    spkr_id, age, genre, accent = l[:4]
    return spkr_id, {
        'age': age,
        'genre': genre,
        'accent': accent,
        'region': ' '.join(l[4:])
        }


def _files_in(dirname, extension):
    """Returns list of files in given directory,
    eventually filter by extension.
    """
    file_paths = [os.path.join(dirname, f) for f in os.listdir(dirname)]
    filt = lambda path: (
        os.path.isfile(path)
        and os.path.splitext(path)[-1].lower()[1:] == extension
        )
    return filter(filt, file_paths)


def _speaker_names_from(path):
    """Get a list of speakers from directory names in path.
    """
    directories = filter(lambda p: os.path.isdir(os.path.join(path, p)),
                         os.listdir(path))
    return [d[1:] for d in directories if d[0] == 'p']


def _records_list_from(path, ext):
    try:
        files = _files_in(path, ext)
    except OSError:
        return []
    return [f.split('_')[-1].split('.')[0] for f in files]


class VCTKDB(DataBase):

    WAV_DIR = 'wav48'
    TXT_DIR = 'txt'
    SPKR_FILE = 'speaker-info.txt'

    def __init__(self):
        DataBase.__init__(self)

    def from_db_root(self, root, speakers='sound'):
        """
        @param version: Acorns version, 1 for year 1, 2 for year 2 (default)
        @param speakers: sound | info | text | everything (default)
            (has sound | has info | has transcriptions | has everything)
        """
        root = os.path.abspath(root)
        self.root = root
        speakers_info = self._parse_speakers_info()
        speakers_in_wav = _speaker_names_from(os.path.join(self.root,
                                                           self.WAV_DIR))
        speakers_in_txt = _speaker_names_from(os.path.join(self.root,
                                                           self.TXT_DIR))
        if speakers == 'sound':
            speakers = speakers_in_wav
        elif speakers == 'text':
            speakers = speakers_in_txt
        elif speakers == 'info':
            speakers = speakers_info.keys()
        elif speakers == 'everything':
            speakers = set(speakers_info.keys).intersection(
                set(speakers_in_wav)).intersection(set(speakers_in_txt))
        else:
            raise ValueError
        for s in sorted(speakers):
            spkr_id = self.add_speaker(s, speakers_info.get(s, None))
            # Enumerate wavs and / or transcriptions
            snd_records = _records_list_from(self.get_wav_dir(spkr_id), 'wav')
            txt_records = _records_list_from(self.get_txt_dir(spkr_id), 'txt')
            for r in sorted(set(snd_records).union(txt_records)):
                self._build_record(r, spkr_id)

    def _build_record(self, name, speaker_id):
        prefix = self._get_speaker_dir(speaker_id) + '_' + name
        txt_path = os.path.join(self.get_txt_dir(speaker_id), prefix + '.txt')
        try:
            with open(txt_path, 'r') as f:
                transcription = f.read()
        except IOError:
            transcription = None
        audio = prefix + '.wav'
        full_audio_path = os.path.join(self.get_wav_dir(speaker_id), audio)
        if not os.path.exists(full_audio_path):
            audio = None
        tags = []  # TODO: eventually extract words / radicals
        r = Record(self, speaker_id, audio, tags, transcription, STYLE)
        self.add_record(r)

    def _get_speaker_dir(self, speaker_id):
        return 'p' + self.spkrs[speaker_id]

    def get_wav_dir(self, speaker_id):
        return os.path.join(self.root, self.WAV_DIR,
                            self._get_speaker_dir(speaker_id))

    def get_txt_dir(self, speaker_id):
        return os.path.join(self.root, self.TXT_DIR,
                            self._get_speaker_dir(speaker_id))

    def _parse_speakers_info(self):
        with open(os.path.join(self.root, self.SPKR_FILE), 'r') as f:
            return dict([_parse_speaker_info(l) for l in f.readlines()])

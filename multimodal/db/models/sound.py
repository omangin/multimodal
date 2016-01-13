# -*- coding: utf-8 -*-


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '10/2011'


import os
import json


class Record:

    def __init__(self, db, speaker, audio, tags,
                 transcription, style):
        self.db = db
        self.spkr_id = speaker
        self.audio = audio  # Name of audio file
        self.tags = tags  # tag indices
        self.trans = transcription
        self.style = style

    def __lt__(self, other):
        assert isinstance(other, Record)
        return self.audio < other.audio

    def __le__(self, other):
        assert isinstance(other, Record)
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __str__(self):
        return "<\"{}\", {}, tags: {} ({})>".format(
            self.trans,
            self.audio,
            [self.db.tags[t] for t in self.tags],
            self.db.spkrs[self.spkr_id],
            )

    def __repr__(self):
        return self.__str__()

    def get_tag_names(self):
        return [self.db.tags[i] for i in self.tags]

    def get_audio_path(self):
        return os.path.join(self.db.get_wav_dir(self.spkr_id), self.audio)

    def to_dict(self):
        return {'audio_file': self.audio,
                'tags': self.get_tag_names(),
                'transcription': self.trans,
                'style': self.style,
                }

    def getDom(self, doc):
        rd = doc.createElement('record')
        rd.setAttribute('style', self.style)
        # Store audio file
        audio = doc.createElement('audio')
        audio.appendChild(doc.createTextNode(self.audio))
        rd.appendChild(audio)
        # Store transcription
        trans = doc.createElement('trans')
        trans.appendChild(doc.createTextNode(self.trans))
        rd.appendChild(trans)
        # Store tags
        for t in self.get_tag_names():
            tag = doc.createElement('tag')
            tag.setAttribute('name', t)
            rd.appendChild(tag)
        return rd

    @classmethod
    def from_dict(cls, db, speaker_id, d):
        return cls(db, speaker_id, d.get('audio_file', None),
                   [db.get_tag_add(t) for t in d.get('tags', [])],
                   d.get('transcription', None), d.get('style', None))


class DataBase:

    WAV_DIR = ''  # Directory for wav files

    def __init__(self):
        # Init record list, organized by speaker
        self.records = []
        # Init tag list:
        #   tags are strings to which an index is associated
        #   from their order of addition
        self.tags = []
        self.tag_id = {}
        self.root = None
        self.spkrs = []
        self.spkrs_info = []

    def has_tag(self, tag):
        return tag in self.tag_id

    def add_tag(self, tag):
        if self.has_tag(tag):
            raise ValueError("Existing tag: %s" % tag)
        self.tags.append(tag)
        tagid = len(self.tags) - 1
        self.tag_id[tag] = tagid
        return tagid

    def get_tag_add(self, tag):
        """Return tag id and creates it if necessary.
        """
        if not self.has_tag(tag):
            tagid = self.add_tag(tag)
        else:
            tagid = self.tag_id[tag]
        return tagid

    def sort(self):
        for r in self.records:
            r.sort()

    def get_wav_dir(self, speaker_id):
        return os.path.join(self.root, self.WAV_DIR,
                            self.spkrs[speaker_id])

    def write_json(self, filename):
        data = {'root': self.root,
                'tags': self.tags,
                'speakers': []
                }
        for speaker, info, records in zip(self.spkrs, self.spkrs_info,
                                          self.records):
            speaker_records = [r.to_dict() for r in records]
            data['speakers'].append({
                'name': speaker,
                'info': info,
                'records': speaker_records,
                })
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from(self, filename, sort=True):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.root = data['root']
        for t in data.get('tags', []):
            self.add_tag(t)
        for s in data['speakers']:
            spk_id = self.add_speaker(s['name'], s.get('info', None))
            for r in s['records']:
                r = Record.from_dict(self, spk_id, r)
                self.records[-1].append(r)
        if sort:
            self.sort()

    def add_speaker(self, name, info=None):
        if name in self.spkrs:
            raise ValueError('There is already a speaker with the same name.')
        self.records.append([])
        self.spkrs.append(name)
        self.spkrs_info.append(info)
        return len(self.spkrs) - 1

    def add_record(self, record):
        if record.db is not self:
            raise ValueError("Record belongs to another db.")
        for t in record.tags:
            if t > len(self.tags):
                raise ValueError("Record contains invalid tags.")
        self.records[record.spkr_id].append(record)

    def size(self):
        return sum(map(len, self.records))

    def __str__(self):
        return ("%s records for %s speakers and %s keywords"
                % (self.size(), len(self.records), len(self.tags),))

    def count_by_keywords(self):
        nb_kw = len(self.tags)
        counts = [[0 for _ in range(nb_kw)] for __ in range(nb_kw)]
        for s in self.records:
            for r in s:
                for t in r.tags:
                    for u in r.tags:
                        if t != u:
                            counts[t][u] += 1
        return counts

    def all_records(self):
        for s in self.records:
            for r in s:
                yield r

    def statistics(self, display=True):
        print("=======================")
        print("* Database statistics *")
        print("=======================")
        print(self.__str__())
        print("Records by speaker: %s" % ", ".join(map(len, self.records)))
        counts = [0 for _ in self.tags]
        for r in self.all_records():
            for t in r.tags:
                counts[t] += 1
        print("Records by keywords: %s" % ", ".join(zip(self.tags, counts)))

# -*- coding: utf-8 -*-


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '10/2011'


import os
import xml.dom.minidom as dom


XML_DIR = 'XML'
WAV_DIR = 'WAV'
WAV_EXT = '.wav'


def get_text_node(dom, el_name):
    txt = dom.getElementsByTagName(el_name)[0].childNodes[0].data
    return txt.strip()


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
        return os.path.join(self.db.root, WAV_DIR,
                self.db.spkrs[self.spkr_id], self.audio)

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


class DataBase:

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

    def todom(self):
        doc = dom.Document()
        db = doc.createElement('database')
        # Add root description
        root = doc.createElement('root')
        root.appendChild(doc.createTextNode(self.root))
        db.appendChild(root)
        # Add tag declarations
        for tagname in self.tags:
            tag = doc.createElement('tag_def')
            tag.setAttribute('name', tagname)
            db.appendChild(tag)
        # Add speaker descriptions and records
        for (spkr, recs) in zip(self.spkrs, self.records):
            s = doc.createElement('speaker')
            s.setAttribute('dir', str(spkr))
            for r in recs:
                s.appendChild(r.getDom(doc))
            db.appendChild(s)
        doc.appendChild(db)
        return doc

    def write_xml(self, filename):
        with open(filename, 'w') as f:
            f.write(self.todom().toprettyxml(indent='    '))

    def load_from(self, filename, sort=True):
        parsed = dom.parse(filename)
        db = parsed.getElementsByTagName('database')[0]
        new_root = get_text_node(db, 'root')
        if self.root is not None and self.root != new_root:
            raise ValueError(
                    "Can't load records from a DB with a different root.")
        else:
            self.root = new_root
        # Get tags
        tags = db.getElementsByTagName('tag_def')
        for tag in tags:
            self.add_tag(tag.getAttribute('name'))
        # Get speakers
        speakers = db.getElementsByTagName('speaker')
        for speaker in speakers:
            spk_id = len(self.spkrs)
            self.spkrs.append(speaker.getAttribute('dir'))
            self.records.append([])
            # Get records
            records = speaker.getElementsByTagName('record')
            for rec in records:
                style = rec.getAttribute('style')
                audio = get_text_node(rec, 'audio')
                tags = [self.tag_id[t.getAttribute('name')]
                        for t in rec.getElementsByTagName('tag')]
                trans = get_text_node(rec, 'trans')
                r = Record(self, spk_id, audio, tags, trans, style)
                self.records[-1].append(r)
        if sort:
            self.sort()

    def add_speaker(self, name):
        self.records.append([])
        self.spkrs.append(name)

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
        print "======================="
        print "* Database statistics *"
        print "======================="
        print self.__str__()
        print("Records by speaker: %s" % ", ".join(map(len, self.records)))
        counts = [0 for _ in self.tags]
        for r in self.all_records():
            for t in r.tags:
                counts[t] += 1
        print("Records by keywords: %s" % ", ".join(zip(self.tags, counts)))

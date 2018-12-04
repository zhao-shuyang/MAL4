#!/usr/bin/env python
import numpy as np
import sqlite3
import marshal, pickle
import collections
from datetime import datetime

class LabelSet(object):
    """
    Audio Segment dataset managing with SQL.
    """
    def __init__(self, file_name):
        self.label_type_map = {'ground_truth':0, 'prediction':1, 'annotation':2}
        self.__connect__(file_name)

    def __connect__(self, file_name):
        #Connect to database
        self.db_conn = sqlite3.connect(file_name)        
        self.db_conn.text_factory = str
        self.cursor=self.db_conn.cursor()
        self.db_conn.execute("PRAGMA foreign_keys=ON")
        self.db_conn.execute("PRAGMA busy_timeout = 300000")
        self.db_conn.text_factory = bytes
        
    def __close__(self):
        self.db_conn.close()

    def __create_tables__(self):
        sql = """
        CREATE TABLE IF NOT EXISTS segments(
        segment_id VARCHAR(32) PRIMARY KEY,
        audio_file VARCHAR(256),
        feature_start_index INTEGER,
        feature_end_index INTEGER,
        repr_index INTEGER,        
        UNIQUE (audio_file) ON CONFLICT IGNORE
        );
        """
        self.db_conn.execute(sql)
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS segment_indexing ON segments (segment_id);")
        sql = """
        CREATE TABLE IF NOT EXISTS classes(
        class_id INTEGER PRIMARY KEY,
        ASID VARCHAR(32) UNIQUE ON CONFLICT IGNORE, 
        class_name VARCHAR(256) UNIQUE ON CONFLICT IGNORE 
        );
        """
        self.db_conn.execute(sql)
        
        sql = """
        CREATE TABLE IF NOT EXISTS labels(
        label_id INTEGER PRIMARY KEY,
        segment_id INTEGER NOT NULL,
        class_id INTEGER NOT NULL,
        label_type INTEGER NOT NULL, 
        FOREIGN KEY (segment_id) REFERENCES segments(segment_id) ON DELETE CASCADE,
        FOREIGN KEY (class_id) REFERENCES classes(class_id) ON DELETE CASCADE,
        UNIQUE (segment_id, class_id, label_type) ON CONFLICT IGNORE
        );
        """
        self.db_conn.execute(sql)

        self.db_conn.execute("CREATE INDEX IF NOT EXISTS label_seg_index ON labels (segment_id);")
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS label_class_index ON labels (class_id);")
    
    def __commit__(self):
        self.db_conn.commit()

    def __insert__(self, record_dict, table_name):
        """
        record_dict: a dictionary indicating values to be inserted.
        table_name: name of the target table.
        """
        sql = """
        INSERT OR IGNORE INTO {0} ({1}) VALUES ({2})
        """.format(table_name, ','.join(record_dict.keys()), ','.join(["'{0}'".format(str(v)) for v in record_dict.values()]))
        print (sql)
        self.cursor.execute(sql)
        return self.cursor.lastrowid

    def __delete__(self, record_dict, table_name):
        """
        record_dict: a dictionary indicating values to be inserted.
        table_name: name of the target table.
        """
        condition_sql = ' AND ' .join(["{0}='{1}'".format(k,v) for k,v in list(record_dict.items())]) 
        sql = """
        DELETE FROM {0}
        WHERE {1}
        """.format(table_name, condition_sql)

        self.cursor.execute(sql)
    
    def initialize(self):
        self.__create_tables__()

    def clear_labels(self, label_type='annotation', exception_list=[]):
        if label_type in ['ground_truth', 'prediction', 'annotation']:
            label_type = self.label_type_map[label_type]
        if exception_list:
            exception_cond = "AND segment_id NOT IN ({0})".format(','.join([str(segment_id) for segment_id in exception_list]))
            #print exception_cond
        else:
            exception_cond = ''
            sql = """
            DELETE  FROM labels
            WHERE label_type = {0}
            {1}
            """.format(label_type, exception_cond)
        self.cursor.execute(sql)
        
    def get_segment_by_id(self, segment_id):
        sql = """
        SELECT audio_file, feature_index FROM segments
        WHERE segment_id == {0}
        """.format(segment_id)
        self.cursor.execute(sql)
        return 

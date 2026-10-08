import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from dashboard import attachment_audit as a


class AttachmentArtifactChecks(unittest.TestCase):
    def test_completed_export_only_and_declared_buffer_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'attachment_audit/camel';out.mkdir(parents=True)
            manifest={'schema':1,'sequence':'camel','complete':False,'geometry':{'labels':{'file':'labels.bin','dtype':'int8','shape':[3]}}}
            def write(name,data):
                path=out/name;path.write_bytes(data);os.utime(path,(time.time()-5,time.time()-5))
            write('audit.json',json.dumps(manifest).encode());write('labels.bin',b'\x02\x03\x04')
            self.assertEqual(a.exports(root),[])
            with self.assertRaises(ValueError):a.asset(root,'attachment_audit/camel','labels.bin')
            manifest['complete']=True;write('audit.json',json.dumps(manifest).encode())
            self.assertEqual(len(a.exports(root)),1)
            self.assertEqual(a.asset(root,'attachment_audit/camel','labels.bin'),b'\x02\x03\x04')
            with self.assertRaises(ValueError):a.asset(root,'attachment_audit/camel','../labels.bin')
            write('labels.bin',b'\x02')
            with self.assertRaises(ValueError):a.asset(root,'attachment_audit/camel','labels.bin')


if __name__=='__main__':unittest.main()

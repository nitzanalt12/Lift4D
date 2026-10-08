"""Local HTTP UI. Reads saved artifacts only; writes exclusively to its cache."""
import argparse
import io
import importlib.metadata
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np
from PIL import Image
from . import artifacts as a
from .catalog import build as result_catalog
from .comparison import pair as compare_views
from .metrics import boundary, cached_evaluate, local_lpips

STATIC = Path(__file__).parent/'static'
EVAL_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def respond(self, data, mime='application/json', status=200):
        if mime == 'application/json':
            data = json.dumps(data,allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            url = urlparse(self.path)
            q = {k:v[0] for k,v in parse_qs(url.query).items()}
            if url.path in ['/', '/app.js', '/comparison.js', '/style.css']:
                name = 'index.html' if url.path=='/' else url.path[1:]
                mime = {'index.html':'text/html; charset=utf-8','app.js':'text/javascript','comparison.js':'text/javascript','style.css':'text/css'}[name]
                return self.respond((STATIC/name).read_bytes(),mime)
            if url.path=='/api/catalog':
                return self.respond(result_catalog(self.server.root,q.get('auxiliary')=='1'))
            if url.path=='/api/runs':
                return self.respond({'runs':a.runs(self.server.root),'root':str(self.server.root)})
            if url.path=='/api/run':
                return self.respond(a.describe(self.server.root,q['run']))
            run = a.inside(self.server.root,q['run'])
            if url.path=='/api/compare':
                return self.respond(compare_views(run,a.inside(self.server.root,q['run_b']),q['sequence'],q.get('view'),q.get('view_b')))
            if url.path=='/api/views':
                return self.respond({'views':a.views(run,q['sequence'])})
            records, view, manifest = a.frame_index(run,q['sequence'],q.get('view'))
            if url.path=='/api/frames':
                mask_values = []
                try:
                    if records:
                        mask_values = [int(v) for v in np.unique(a.image(records[0]['target'])) if v != 0]
                except (ValueError, OSError):
                    pass
                return self.respond({'frames':[{'id':r['id'],'time':r['time'],'render_available':bool(r['render']),
                                                'reason':r['reason'],'aligned':r['alignment']} for r in records],
                                     'view':view,'manifest':manifest,'target_mask_values':mask_values,
                                     'target_source': records[0]['target'] if records else None})
            if url.path=='/api/image':
                rec = next(r for r in records if r['id']==q['frame'])
                obj = int(q.get('object','1'))
                threshold = float(q.get('threshold','0.5'))
                blend = float(q.get('blend','0.5'))
                if not 0<=threshold<=1 or not 0<=blend<=1:
                    raise ValueError('Threshold and blend must be within [0,1]')
                kind=q.get('kind','input')
                if kind=='input':
                    pixels=a.image(rec['input'],'RGB')
                else:
                    t,r,g,alpha=a.arrays(rec,obj)
                    if r is None:
                        raise ValueError('Saved render is not available')
                    if kind=='render':
                        pixels=r
                    elif kind=='overlay':
                        pixels=np.round(t*(1-blend)+r*blend).astype(np.uint8)
                        pixels[boundary(g)]=[20,230,255]
                        if alpha is not None:
                            pixels[boundary(alpha>=threshold)]=[255,70,160]
                    else:
                        raise ValueError('Unknown image kind')
                out=io.BytesIO();Image.fromarray(pixels).save(out,format='PNG')
                return self.respond(out.getvalue(),'image/png')
            self.respond({'error':'Not found'},status=404)
        except (ValueError, OSError, KeyError, StopIteration, TypeError) as e:
            self.respond({'error':str(e)},status=422)

    def do_POST(self):
        try:
            if self.path!='/api/evaluate':
                return self.respond({'error':'Not found'},status=404)
            size=int(self.headers.get('Content-Length','0'))
            if size>16384:
                raise ValueError('Request too large')
            q=json.loads(self.rfile.read(size))
            threshold=float(q.get('threshold',0.5));obj=int(q.get('object',1))
            if not 0<=threshold<=1 or obj<1:
                raise ValueError('Invalid evaluation settings')
            run=a.inside(self.server.root,q['run'])
            records,_,manifest=a.frame_index(run,q['sequence'],q.get('view'))
            settings={'alpha_threshold':threshold,'object_id':obj,'target_source':'DAVIS indexed PNG label equality',
                      'sequence':q['sequence'],'lpips':False,'appearance_roi':'target mask; background=0.5 for spatial LPIPS',
                      'boundary':'pooled bidirectional 8-neighbor inner boundary Euclidean distances, mean and p95',
                      'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','Pillow']}}
            with EVAL_LOCK:
                fn=None; note=None
                if q.get('lpips') and any(r['alignment'] for r in records):
                    try:
                        fn, fingerprint=local_lpips();settings['lpips']=fingerprint
                    except (ImportError,OSError,ValueError,RuntimeError) as e:
                        note=str(e)
                result=cached_evaluate(records,settings,self.server.cache,fn)
            result['lpips_note']=note
            self.respond(result)
        except (ValueError,OSError,KeyError,TypeError) as e:
            self.respond({'error':str(e)},status=422)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',default='runs')
    p.add_argument('--cache-root',default=str(Path.home()/'.cache/lift4d-dashboard'))
    p.add_argument('--port',type=int,default=8765)
    args=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.root=Path(args.runs_root).resolve();server.cache=Path(args.cache_root).resolve()
    print(f'LIFT4D dashboard: http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()

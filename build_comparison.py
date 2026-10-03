"""Generate a self-contained CNN / ViT explorer with trained weights."""
import argparse,base64,io,json,sys,hashlib
from pathlib import Path
import torch
from PIL import Image

def encode(x):
    return base64.b64encode(x.detach().cpu().numpy().astype('<f4').tobytes()).decode()
def png(im):
    b=io.BytesIO();im.save(b,format='PNG');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
def maps(x):
    x=x[0].detach().cpu();mx=float(x.max());v=(x/(mx or 1)*255).round().byte()
    return {'shape':list(x.shape),'maximum':mx,'maps':[png(Image.fromarray(a.numpy())) for a in v]}
def vit_capture(model,x):
    t=model.patches(x).flatten(2).transpose(1,2)
    t=torch.cat((model.class_token.expand(x.shape[0],-1,-1),t),dim=1)+model.position
    states=[t];attention=[]
    for layer in model.encoder.layers:
        z=layer.norm1(t)
        a,weights=layer.self_attn(z,z,z,need_weights=True,average_attn_weights=False)
        t=t+a
        t=t+layer.linear2(layer.activation(layer.linear1(layer.norm2(t))))
        attention.append(weights[0]);states.append(t)
    logits=model.head(model.norm(t[:,0]))
    torch.testing.assert_close(logits,model(x),rtol=2e-4,atol=2e-5)
    return {'states':[encode(v[0]) for v in states],'attention':[encode(v) for v in attention],'probabilities':logits.softmax(-1)[0].tolist()},[v[0].tolist() for v in states],logits[0].tolist()
def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path('/Users/daryn/Desktop/midterm_project 2'));a=p.parse_args()
    sys.path.insert(0,str(a.project));import cnn_model,vit_model
    torch.set_num_threads(4)
    registry=json.loads((a.project/'models/registry.json').read_text())
    infos={kind:next(v for v in registry.values() if v['type']==kind) for kind in ['cnn','vit']}
    assert infos['cnn']['classes']==infos['vit']['classes']
    models={'cnn':cnn_model.build_model().eval(),'vit':vit_model.build_model().eval()}
    data={'classes':infos['cnn']['classes'],'weights':{},'metrics':{},'samples':[],'hashes':{}}
    for kind,m in models.items():
        path=a.project/infos[kind]['file'];m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        data['hashes'][kind]=hashlib.sha256(path.read_bytes()).hexdigest()
        data['weights'][kind]={k:encode(v) for k,v in m.state_dict().items()}
        met=json.loads((a.project/f'results/{kind}_metrics.json').read_text())
        data['metrics'][kind]={k:met[k] for k in ['test_accuracy','test_macro_f1','split_counts','correct_count']}
        data['metrics'][kind]['parameters']=sum(p.numel() for p in m.parameters())
    with torch.inference_mode():
        for path in sorted((a.project/'test_images').glob('*.png')):
            with Image.open(path) as im:x=cnn_model.prepare_image(im).unsqueeze(0)
            sample={'name':path.stem,'input':png(Image.fromarray((x[0].permute(1,2,0).numpy()*255).round().astype('uint8')))}
            y=x;stages=[]
            for i,layer in enumerate(models['cnn']):
                y=layer(y)
                if i in [3,7,11,15,16]:stages.append(maps(y))
            torch.testing.assert_close(y,models['cnn'](x))
            sample['cnn']={'stages':stages,'probabilities':y.softmax(-1)[0].tolist()}
            sample['vit'],_,_=vit_capture(models['vit'],(x-.5)/.5)
            data['samples'].append(sample)
        # Full reference tensors for independent browser inference validation.
        x=(torch.arange(3*96*96)%251).float().reshape(1,3,96,96)/250
        cap,states,logits=vit_capture(models['vit'],x*2-1)
        reference={'states':states,'attention':cap['attention'],'logits':logits,'cnn':models['cnn'](x)[0].tolist()}
    root=Path(__file__).parent
    template=(root/'comparison-template.html').read_text().replace('__ENGINE__',(root/'inference.js').read_text()).replace('__VIEWER__',(root/'viewer.js').read_text())
    payload=json.dumps(data,separators=(',',':')).replace('<','\\u003c')
    (root/'cnn-vit.html').write_text(template.replace('__MODEL_DATA__',payload))
    # Reference tensors are used during development; the generated viewer includes model data only.
    print('Built comparison with',len(data['samples']),'samples. CNN and ViT captured outputs match normal forward passes.')
if __name__=='__main__':main()

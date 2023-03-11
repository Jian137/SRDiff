import torch
m = torch.load('/home/vipsl416-4-zhanglize/NAS/Code/SISR/SRDiff/checkpoints/DASr/model_600.pt')
print(m['E.encoder_q.E.1.running_var'])
c = dict()
c['denoise_fn.E.queue_ptr'] = m['E.queue_ptr']
for k in m.keys():
    if k.startswith('G.'):
        continue
    c["denoise_fn.E.encoder_q"+k[11:]] = m[k]
c.pop('denoise_fn.E.encoder_q')
for k in m.keys():
    if k.startswith('G.'):
        continue
    c["denoise_fn.E.encoder_k"+k[11:]] = m[k]
c.pop('denoise_fn.E.encoder_k')

torch.save(c,"zzz.pt")

k = torch.load('zzz.pt')
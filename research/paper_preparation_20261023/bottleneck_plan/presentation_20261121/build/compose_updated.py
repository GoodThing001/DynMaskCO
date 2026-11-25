from pathlib import Path

b=Path(__file__).resolve().parent
old=b.parent.parent/'presentation_20261120'/'build'
s=(old/'build_compact_deck.mjs').read_text(encoding='utf-8')
s=s[:s.index("{\n const a=add('数据来源与版本'")]
s=s.replace("from './","from '../../presentation_20261120/build/")
s="import {addEvidenceSlides} from './updated_content.mjs';\n"+s
s=s.replace("const DIR=BASE+'/presentation_20261120';","const DIR=BASE+'/presentation_20261120';\nconst NEXT=BASE+'/presentation_20261121';\nconst E=JSON.parse(await fs.readFile(NEXT+'/build/latest_evidence.json','utf8'));")
s=s.replace('2026年11月20日','2026年11月21日').replace("'模型设计与实验结果'","'模型设计与统一前瞻实验'")
s+='''
await addEvidenceSlides({...ctx,add,note,stat,chartOwners,tableOwners,P},E);
if(P.slides.items.length!==20)throw new Error('Expected 20 slides, got '+P.slides.items.length);
const script=await fs.readFile(BASE+'/model_documentation_20261121/build/presentation_script.md','utf8');
const speeches=[...script.matchAll(/^## 第 (\\d+) 页[^\\n]*\\n([\\s\\S]*?)(?=^## |$(?![\\s\\S]))/gm)];
if(speeches.length!==20)throw new Error('Expected 20 corresponding speech sections');
for(const m of speeches)P.slides.getItem(Number(m[1])-1).speakerNotes.append('\\n\\n'+m[2].replaceAll('**','').trim());
await fs.writeFile(NEXT+'/build/slide_manifest.json',JSON.stringify({slideList,chartOwners:[...new Set(chartOwners)],tableOwners:[...new Set(tableOwners)]},null,2));
await (await PresentationFile.exportPptx(P)).save(NEXT+'/build/candidate.pptx');
console.log('Exported',P.slides.items.length);
'''
(b/'build_deck.mjs').write_text(s,encoding='utf-8')

final=(old/'finalize_compact.mjs').read_text(encoding='utf-8')
final=final.replace("presentation_20261120';","presentation_20261121';")
final=final.replace('slide_manifest_compact.json','slide_manifest.json').replace('candidate_compact_final_complete.pptx','candidate_complete.pptx')
final=final.replace('MASKCO_静态冷链模型汇报_20页完整版_20261120.pptx','MASKCO_静态冷链模型汇报_统一前瞻实验_20261121.pptx').replace('final_compact_delivery_ready.validation.json','final.validation.json')
(b/'finalize_deck.mjs').write_text(final,encoding='utf-8')
embed=(old/'embed_compact_chart_data.py').read_text(encoding='utf-8').replace('candidate_compact_final.pptx','candidate.pptx').replace('candidate_compact_final_complete.pptx','candidate_complete.pptx').replace('chart_data_compact_final_ready.json','chart_data_ready.json')
(b/'embed_chart_data.py').write_text(embed,encoding='utf-8')
print('Prepared template-preserving builder')

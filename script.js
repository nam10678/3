// script.js (load model from ./model/model.json)
let history = [];
let model = null;
const mapToNum = {'P':0,'B':1,'T':2};
const numToLabel = ['P','B','T'];
const colorMap = ['#2196F3','#f44336','#9E9E9E'];

async function loadModel(){
  try{
    model = await tf.loadLayersModel('model/model.json');
    console.log('Model loaded');
    document.getElementById('prediction')?.removeAttribute?.('data-loading');
  }catch(e){
    console.error('Lỗi tải mô hình', e);
    let p = document.getElementById('prediction');
    if(p) p.innerText = '⚠️ Lỗi tải mô hình: ' + (e.message||e);
  }
}
loadModel();

function addResult(r){
  if(history.length >= 100) history.shift();
  history.push(r);
  renderHistory();
  if(history.length >= 5) predictAuto();
}

function resetHistory(){
  history = [];
  renderHistory();
  clearPrediction();
}

function renderHistory(){
  document.getElementById('history').innerText = history.join(' ');
}

function clearPrediction(){
  document.getElementById('predicted-label').innerText = '--';
  document.getElementById('predicted-label').style.background = 'transparent';
  document.getElementById('confidence').innerText = '--';
}

// one-hot encode last 5 -> 15 features
function encodeOneHot5(seq){
  const out = [];
  const s = Array.from(seq);
  while(s.length < 5) s.unshift('P'); // pad front if needed
  for(let i=0;i<5;i++){
    const v = s[i];
    const n = mapToNum[v];
    out.push(n===0?1:0);
    out.push(n===1?1:0);
    out.push(n===2?1:0);
  }
  return out;
}

async function predictAuto(){
  if(!model){ console.warn('Model chưa load'); return; }
  const last5 = history.slice(-5);
  if(last5.length < 5) return;
  try{
    const input = encodeOneHot5(last5);
    const tensor = tf.tensor2d([input], [1, 15], 'float32');
    const out = model.predict(tensor);
    const probs = await out.data();
    const maxIdx = probs.indexOf(Math.max(...probs));
    const label = numToLabel[maxIdx];
    const conf = (probs[maxIdx]*100).toFixed(2);
    const labEl = document.getElementById('predicted-label');
    labEl.innerText = label;
    labEl.style.background = colorMap[maxIdx];
    document.getElementById('confidence').innerText = conf + '%';
  }catch(err){
    console.error('Predict error', err);
    let p = document.getElementById('prediction');
    if(p) p.innerText = '⚠️ Lỗi dự đoán: ' + (err.message||err);
  }
}

"""INTABEH — Streamlit evaluation wrapper. Frozen inference files are unchanged.
This is a record-and-analyze trial, NOT continuous background detection.
"""
import os, sys, io, hashlib, json
from pathlib import Path
import streamlit as st
import soundfile as sf
import numpy as np
ROOT=Path(__file__).resolve().parent
PKG=ROOT/'05_AI_MODEL'/'03_MODELS'/'FINAL_v1.0'
os.environ['TFHUB_CACHE_DIR']=str(ROOT/'tfhub_cache')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_NUM_INTEROP_THREADS']='1'
os.environ['TF_NUM_INTRAOP_THREADS']='1'
sys.path.insert(0,str(ROOT/'07_WEB_APP'/'source'/'backend'))
from intabeh_service import preprocess, live_clip_probs, LABELS_AR, lockimport intabeh_inference as inf
st.set_page_config(page_title='انتبه | INTABEH',page_icon='🔔',layout='centered')
st.markdown('<div dir="rtl" style="text-align:right"><h1>🔔 انتبه | INTABEH</h1><p>تجربة نشر للنموذج الأساسي المعتمد Core V1</p></div>',unsafe_allow_html=True)
st.info('نسخة اختبار للنشر: سجّل مقطعًا ثم اضغط تحليل. ليست مراقبة صوتية مستمرة ولا بديلًا لأنظمة السلامة.')
@st.cache_resource(show_spinner='جارٍ تحميل النموذج الأساسي...')
def load_model():
    record=json.loads((PKG/'FREEZE.json').read_text(encoding='utf-8'))
    for name,expected in record['files_sha256'].items():
        actual=hashlib.sha256((PKG/name).read_bytes()).hexdigest()
        if actual!=expected: raise RuntimeError('فشل التحقق من سلامة Core V1: '+name)
    model=inf.IntabehModel(PKG)
    model.clip_probs(np.zeros(inf.WIN,np.float32))
    return model
try:
    model=load_model()
    st.success('Core V1 جاهز — تحققنا من بصمات الملفات المعتمدة')
except Exception as e:
    st.error('تعذر تحميل النموذج. راجعي سجلات النشر.')
    st.code(f'{type(e).__name__}: {e}')
    st.stop()
audio=st.audio_input('سجّلي صوتًا للاختبار (0.3 إلى 10 ثوانٍ)')
if audio is not None:
    st.audio(audio)
    if st.button('تحليل الصوت',type='primary'):
        try:
            raw=audio.getvalue()
            if len(raw)>4*1024*1024: raise ValueError('الملف أكبر من 4MB')
            data,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True)
            duration=len(data)/sr
            if not (16000<=sr<=192000 and 0.3<=duration<=10): raise ValueError('المدة أو معدل العينات غير مدعوم')
            y=preprocess(data,sr)
            if y is None:
                st.info('صوت آخر / خلفية — لم يُكتشف نشاط صوتي كافٍ')
            else:
                with lock:
                    p,k,lv=live_clip_probs(model,y)
                    idx=model.decide(p,'thresholds')
                key=model.labels[idx]
                st.metric('الصوت المتوقع',LABELS_AR.get(key,key))
                st.write('الثقة:',f'{float(p[idx])*100:.1f}%')
                st.caption('نتيجة تجريبية لا تُستخدم كإنذار سلامة أو تشخيص طبي.')
        except Exception as e:
            st.error(f'تعذر تحليل الصوت: {e}')

import streamlit as st 
import pandas as pd 
import plotly.express as px
import yfinance as yf
from tqdm import tqdm
import plotly.graph_objects as go

# ex: streamlit run F:\Drive\Projetos_python\Dash_acoes\dash_monitoring.py
st.set_page_config(layout='wide')

# ----------------------------- FUNCOES e CLASSES ------------------------------------------------

def formata_numero(valor, prefixo =''):
    for unidade in ['', 'mil']:
        if valor < 1000:
            return f'{prefixo} {valor:.2f} {unidade}'

        valor /=1000
    return f'{prefixo} {valor:.2f} milhões'



def get_price_acction(acao_cod, start, end, interval='1d'):
    
    tick = yf.Ticker(acao_cod+'.SA' if acao_cod!='BVSP' else '^BVSP')
    df_aux = tick.history(
        start=start,
        end=end,
        interval=interval,  # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        actions=False
    )
    
    # variaacoes
    df_aux['delta'] = df_aux['Close'].pct_change()
    df_aux['rendimento'] = df_aux['delta'].cumprod()
    df_aux['rendimento'] = df_aux['delta']+1
    df_aux['rendimento'] = df_aux['rendimento'].cumprod()
    df_aux['acao'] = acao_cod.split('.')[0]
    return df_aux


def get_price_acctions(acoes, start, end, interval='1d'):
    df = pd.DataFrame()
    for tk in tqdm(acoes):
        df_aux = get_price_acction(tk, start, end, interval)
        df = pd.concat([df, df_aux])

    colunas = df.columns.copy()
    df = df.reset_index()
    df.columns = ['Date']+list(colunas)
    return df

def get_table_plot(df,preco_referencia):
    df_acoes_principais = pd.pivot_table(
        data=df.reset_index(),
        index='Date',
        columns='acao',
        values=preco_referencia)
    
    
    df_acoes_principais_deltas = pd.pivot_table(
        data=df.reset_index(),
        index='Date',
        columns='acao',
        values='rendimento').dropna()
    
    return df_acoes_principais,df_acoes_principais_deltas


# --------------------------------  ENTRADA DOS DADOS  ----------------------------------------


acoes = [
    'OIBR3', 'COGN3', 'PETR4', 'VVAR3', 'CIEL3', 'BBDC4', 'IRBR3', 'MGLU3', 'VALE3', 'BEEF3', 'ITUB4', 'JBSS3',
    'MRFG3', 'ITSA4', 'BRFS3', 'ABEV3', 'BRML3', 'POMO4', 'SUZB3', 'EMBR3', 'USIM5', 'CSNA3', 'BBAS3', 'RAIL3',
    'CVCB3', 'GOAU4', 'GOLL4',
    'AZUL4', 'PRIO3', 'PETR3', 'GGBR4', 'B3SA3', 'STBP3', 'LAME4', 'CMIG4', 'AERI3', 'HAPV3', 'TOTS3', 'LREN3',
    'CCRO3', 'OIBR4', 'JHSF3', 'BRDT3', 'KLBN11', 'BBDC3', 'EQTL3', 'GNDI3', 'MULT3', 'CYRE3', 'TIMS3', 'RENT3',
    'RAPT4', 'QUAL3', 'ELET3', 'CESP6', 'VIVT3', 'HGTX3', 'UGPA3', 'BIDI4', 'WEGE3', 'SBSP3', 'PETZ3', 'MRVE3', 'NTCO3',
    'GMAT3', 'BTOW3', 'CRFB3', 'ODPV3', 'EVEN3', 'BPAC11', 'BMGB4', 'BKBR3', 'ECOR3', 'SULA11', 'MOVI3', 'RADL3',
    'YDUQ3', 'BRKM5', 'BBSE3', 'CSAN3', 'DTEX3', 'RLOG3', 'SANB11', 'SAPR4', 'BPAN4', 'RDOR3', 'ELET6', 'BOAS3', 'MDIA3',
    'TAEE11', 'ENAT3', 'SMLS3', 'CPFE3', 'ENBR3', 'IGTA3', 'LWSA3', 'LCAM3', 'CURY3', 'KLBN4', 'CSMG3', 'CAML3', 'LAME3',
    'SLCE3', 'BRSR6', 'GRND3', 'EZTC3', 'BRPR3', 'SAPR11', 'BRAP4', 'FLRY3', 'MYPK3', 'LIGT3', 'EGIE3', 'ALSO3',
    'NEOE3', 'HBSA3', 'HYPE3', 'TIET11', 'LJQQ3', 'CEAB3', 'TUPY3', 'ALPA4', 'TRPL4', 'ABCB4', 'ENEV3', 'TEND3',
    'SOMA3', 'CNTO3', 'SMTO3', 'ALUP11', 'ENJU3', 'PNVL3', 'TRIS3', 'SEQL3', 'BIDI3', 'ENGI11', 'PARD3', 'ANIM3',
    'GUAR3', 'CPLE6', 'VIVA3', 'UNIP6', 'LINX3', 'PSSA3', 'PCAR3', 'PGMN3', 'CCPR3', 'ARZZ3', 'SIMH3', 'TIET4',
    'CMIG3', 'POMO3', 'USIM3', 'ITUB3', 'OMGE3', 'AMBP3', 'KLBN3', 'LOGG3', 'JPSA3', 'TAEE4', 'LEVE3', 'RRRP3',
    'TIET3', 'FRAS3', 'CPLE3', 'BVSP'
]
acoes.sort()

intervalos = [
    '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
]

# side bar
st.sidebar.title('Acoes')
list_acoes = st.sidebar.multiselect('Acoes', acoes,default='PETR4')
start = st.sidebar.date_input('Data de inicio',value=pd.to_datetime('2023-01-01'))
end = st.sidebar.date_input('Data de fim',value="today")
interval = st.sidebar.selectbox('Intervalo',intervalos,index=8)

preco_referencia=st.sidebar.selectbox('Preço',['Open', 'High', 'Low', 'Close', 'Volume'])
#todos_anos = st.sidebar.checkbox('Dados de todo o período', value=True)

df = get_price_acctions(
        list_acoes, 
        start=start, 
        end=end, 
        interval=interval
        )


# # -------------------------------  TABELAS -------------------------------------

df['Valor_negociado'] = df['Close']*df['Volume']
df_vol = df.groupby('acao').agg(
    {
        'Volume': sum,
        'Valor_negociado': sum
    }
)

import numpy as np


# # ------------------------------- GRÁFICOS ----------------------------------------
def plot_dataframe(df, title="Preço das açoes"):
    '''
    Retorna um objeto grafico plotly para o pd.dataframe df
    '''
    trace1 = {}
    index = pd.to_datetime(df.index.values)#
    for i, k in enumerate(df.columns.to_list()):
        trace1[k] = go.Scatter(x=index,
                               y=df[k].values,
                               name=k)

    fig1 = go.Figure(list(trace1.values()))
    fig1.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'})
    
    return fig1


fig_volume_negociado = px.bar(
    data_frame=df_vol, 
    x=df_vol.index, 
    y='Volume', 
    title='Volume negociado'
    )
fig_valor_negociado = px.bar(
    data_frame=df_vol, 
    x=df_vol.index, 
    y='Valor_negociado', 
    title='Valor negociado'
    )


# ------------------------- VISUALIZACAO -------------------------------------
# # iniciando


st.title('DASH MERCADO FINANCEIRO : ACOMPANHAMENTO')



st.subheader('Ações', divider='rainbow')


# # criand abas:
aba1, aba2= st.tabs(['Única ação', 'Multiplas Ações'])



# # criando columns

# # add metricas
with aba1:
    coluna1, coluna2 = st.columns([1,3])
    with coluna1:
        st.subheader('Acão analisada', divider='rainbow')
        acao_analisys = st.selectbox('Acão', list_acoes)
        start_analisys = st.date_input('Inicio ',value=pd.to_datetime('2023-01-01'))
        end_analisys = st.date_input('Fim ',value="today")
        interval_analisys = st.selectbox('Intervalo modelo',intervalos,index=8)
        kc_plot = st.checkbox('Plot KC')
        
        container = st.container(border=True)
        with container:
            plot_am = st.checkbox('Media movel')
            window_long = st.number_input('Window long',value=28)
            window_short = st.number_input('Window short',value=7)
            center = st.checkbox('Center')
            

    with coluna2:
        st.subheader('KC valores', divider='rainbow')
        # KELTNER CHANNEL CALCULATION
        def get_kc(high, low, close, kc_lookback, multiplier, atr_lookback):
            tr1 = pd.DataFrame(high - low)
            tr2 = pd.DataFrame(abs(high - close.shift()))
            tr3 = pd.DataFrame(abs(low - close.shift()))
            frames = [tr1, tr2, tr3]
            tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
            atr = tr.ewm(alpha=1/atr_lookback).mean()

            kc_middle = close.ewm(kc_lookback).mean()
            kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
            kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr

            return kc_middle, kc_upper, kc_lower
        
        
        df_kc = get_price_acction(acao_analisys, start_analisys, end_analisys, interval_analisys)

        
        # plotando grafico
        def plot_candlestick(df_kc):
            fig = go.Figure(data=[go.Candlestick(x=df_kc.index,
                                                open=df_kc['Open'],
                                                high=df_kc['High'],
                                                low=df_kc['Low'],
                                                close=df_kc['Close'],
                                                increasing_line_color='blue',
                                                decreasing_line_color='red',
                                                name='Acão'
                                                )],
                            )
            fig.update_layout(xaxis_rangeslider_visible=False)
            return fig
        
        fig_candlestick = plot_candlestick(df_kc)
        
        if kc_plot:
            df_kc['kc_middle'], df_kc['kc_upper'], df_kc['kc_lower'] = get_kc(
                df_kc['High'], df_kc['Low'], df_kc['Close'], 20, 2, 10
            )
            fig_candlestick.add_trace(
                go.Scatter(
                    name='Kc Middle',
                    x=df_kc.index,
                    y=df_kc['kc_middle'],
                    mode='lines',
                    line=dict(width=1, dash='dot', color='seagreen')
                )
            )

            fig_candlestick.add_trace(
                go.Scatter(
                    name='Kc Upper',
                    x=df_kc.index,
                    y=df_kc['kc_upper'],
                    mode='lines',
                    marker=dict(color="blue"),
                    line=dict(width=1, dash='dash')
                )
            )

            fig_candlestick.add_trace(
                go.Scatter(
                    name='Kc Lower',
                    x=df_kc.index,
                    y=df_kc['kc_lower'],
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=1, dash='dot'),
                    fillcolor='rgba(0, 0, 255, 0.1)',  # 8, 68, 68
                    fill='tonexty'
                )
            )
        if plot_am:
            df_kc['Close_long'] = df_kc['Close'].rolling(window_long,center=center).mean()
            df_kc['Close_short'] = df_kc['Close'].rolling(window_short,center=center).mean()
            fig_candlestick.add_trace(
                go.Scatter(
                        name='Media_long',
                        x=df_kc.index,
                        y=df_kc['Close_long'],
                        mode='lines',
                        # marker=dict(color="black", size=10, symbol='x'),
                        line=dict(width=1, color='green')
                    )
                )

            fig_candlestick.add_trace(
                go.Scatter(
                    name='Media_short',
                    x=df_kc.index,
                    y=df_kc['Close_short'],
                    mode='lines',
                    # marker=dict(color="cyan", size=10, symbol='x'),
                    line=dict(width=1, color='orange')
                )
            )
            
        st.plotly_chart(fig_candlestick, use_container_width=True)
        
with aba2: 
    coluna1, coluna2 = st.columns([2,1])
    coluna3, coluna4 = st.columns([1,2])
    with coluna1:
        #st.metric('Receita',formata_numero(dados['Preço'].sum(), 'R$'))
        
        df_precos, df_deltas = get_table_plot(df,preco_referencia)

        fig_plot_precos = plot_dataframe(df_precos,'Historico do Preço '+preco_referencia)
        fig_plot_deltas = plot_dataframe(df_deltas,'Rendimento acumulado '+preco_referencia )

        st.plotly_chart(fig_plot_precos, use_container_width=True)
        st.plotly_chart(fig_plot_deltas, use_container_width=True)
        
    with coluna2:
        try:
            st.plotly_chart(fig_volume_negociado, use_container_width=True)
            st.plotly_chart(fig_valor_negociado, use_container_width=True)     
        
        except:
            st.markdown("*Here's a bouquet*")
    
# coding=utf8

import streamlit as st
import base64
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title='Conversor B3 para XLS',layout="centered", initial_sidebar_state = 'auto',
                   page_icon=r'https://www.google.com/url?sa=i&url=http%3A%2F%2Fpixelartmaker.com%2Fart%2F1fbfb4bc3e63545&psig=AOvVaw2iqmc4lGnqqDJp0lT8mExX&ust=1619097098597000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCMiKuu-0j_ACFQAAAAAdAAAAABAV')
# st.beta_set_page_config(page_title='your_title',
#                         page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')
#
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
#
# try:
#   import googleclouddebugger
#   googleclouddebugger.enable(
#     breakpoint_enable_canary=True
#   )
# except ImportError:
#   pass

def download_link(object_to_download, download_filename, download_link_text):
  """
  Generates a link to download the given object_to_download.

  object_to_download (str, pd.DataFrame):  The object to be downloaded.
  download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
  download_link_text (str): Text to display for download link.

  Examples:
  download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
  download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

  """
  if isinstance(object_to_download, pd.DataFrame):
    object_to_download = object_to_download.to_csv(index=False)

  # some strings <-> bytes conversions necessary here
  b64 = base64.b64encode(object_to_download.encode()).decode()

  return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def maybe_mkdir(path):
  if not os.path.isdir(path): os.mkdir(path)

def readNotasB3(file=r"G:\My Drive\Econometrics\Corretoras\XP\20_12.pdf"):
  import tabula
  def parseTopos(topos):
    numNotas = pd.to_numeric([dfTopo.at[0, 'Nr. nota'] if 'Nr. nota' in dfTopo else np.nan for dfTopo in topos])
    datas = pd.to_datetime([dfTopo.at[0, 'Data pregão'] if 'Data pregão' in dfTopo else np.nan for dfTopo in topos],
                           format='%d/%m/%Y')
    return pd.DataFrame(dict(data=datas, nota=numNotas))

  def parseNegocios(df):
    def findCol(col):
      i = np.where(df.columns == col)[0][0]
      while not (isinstance(df.iloc[0, i], str)) and i < len(df.columns):
        i += 1
      return df.columns[i]

    col = findCol('Quantidade')
    df['quantidade'] = df[col].str.replace('.', '').astype(int) * (1 * (df['C/V'] == 'C') - 1 * (df['C/V'] == 'V'))
    df['valorTotal'] = df[findCol('Valor Operação / Ajuste')].str.replace('.', '').str.replace(',', '.').astype(float)
    df['valorUnit'] = (df['valorTotal'] / df['quantidade']).abs()
    df = df[['quantidade', 'valorTotal', 'valorUnit', 'Tipo mercado', 'Prazo', findCol('Especificação do título'),
             'Obs. (*)']]
    df.columns = ['Qtde', 'valorTotal', 'VlrUnit', 'TipoMercado', 'Prazo', 'Ativo', 'DT']
    return df.reset_index(drop=True)

  def paraseRodape(df):
    def parseStrings(v):
      s1, s2, s3 = v[0]
      if ',' in s1.split(' ')[-1]:
        s2 = s1.split(' ')[-1]
      s2 = s2.replace('.', '')
      s2 = s2.replace(',', '.')
      value = pd.to_numeric(s2)

      value *= -1 if s3 == 'D' else 1
      return value

    totalCustos = parseStrings(
      df.iloc[np.where(df['Resumo Financeiro'].str.lower().str.contains('total custos') == True)[0]].values)
    liquidoPara = parseStrings(
      df.iloc[np.where(df['Resumo Financeiro'].str.lower().str.contains('líquido para') == True)[0]].values)
    return totalCustos, liquidoPara

  topos = tabula.read_pdf(file, multiple_tables=True, pages='all', lattice=False, stream=True, area=[52, 350, 68, 561],
                          pandas_options={'dtype': str})

  negocios = tabula.read_pdf(file, multiple_tables=True, pages='all', lattice=False, stream=True,
                             area=[240, 32, 450, 561], pandas_options={'dtype': str})

  rodape = tabula.read_pdf(file, multiple_tables=True, pages='all', lattice=False, stream=True,
                           area=[450, 296, 844, 561], pandas_options={'dtype': str})

  dfNotas = parseTopos(topos)
  diNotas = {}
  for data, nota in dfNotas.drop_duplicates().values:
    if not np.isnan(nota):
      nota = int(nota)
      df = pd.DataFrame([])
      for i in np.where(dfNotas.data == data)[0]:
        df = pd.concat([df, negocios[i]])
      totalCustos, liquidoPara = paraseRodape(rodape[i])
      data = pd.to_datetime(data).date()
      diNotas[nota] = dict(negocios=parseNegocios(df), data=data, totalCustos=totalCustos, liquidoPara=liquidoPara)
  return diNotas

def updateNotasB3(path=r"G:\My Drive\Econometrics/", parseFolder=r"Corretoras\Notas Corretagem\toParse/"):
  files = os.listdir(path + parseFolder)
  files = [file for file in files if file[-3:] == 'pdf']
  dfOp = pd.DataFrame([])
  dfNotas = pd.DataFrame([], columns=['Pregão', 'corretagem', 'liquidoPara'])
  for file in files:
    diNotas = readNotasB3(path + parseFolder + file)
    for nota in diNotas.keys():
      notaStr = str(nota)
      print(f'adding Nota: {notaStr}')
      dfNotas.loc[notaStr] = diNotas[nota]['data'], abs(diNotas[nota]['totalCustos']), diNotas[nota]['liquidoPara']
      df = diNotas[nota]['negocios']
      df['DT'].fillna('', inplace=True)
      dfc = df[df.Qtde > 0].groupby(['VlrUnit', 'Ativo', 'DT']).sum().reset_index()
      dfv = df[df.Qtde < 0].groupby(['VlrUnit', 'Ativo', 'DT']).sum().reset_index()
      dfc['notaNr'] = notaStr
      dfv['notaNr'] = notaStr
      dfOp = pd.concat([dfOp, dfc])
      dfOp = pd.concat([dfOp, dfv])
  if len(dfOp):
    dfOp.reset_index(drop=True, inplace=True)
    dfOp['Oper'] = dfOp.apply(lambda x: 'Compra' if x.Qtde > 0 else 'Venda', axis=1)
    dfOp['DT'] = dfOp.apply(lambda x: 'DT' if 'D' in x.DT else '', axis=1)
    # try to find corresponding asset (ativos)

    # for ativo in dfOp.Ativo.drop_duplicates():
    #   df = self.df_b3inst[self.df_b3inst.CrpnNm.str.contains(ativo.split(' ')[0])]
    #   if len(df):
    #     if 'ON' in ativo: s=df[df.SpcfctnCd.str.contains('ON')].index
    #     elif 'PN' in ativo: s=df[df.SpcfctnCd.str.contains('PN')].index
    #     elif 'UNIT' in ativo: s=df[df.SpcfctnCd.str.contains('UNIT')].index
    #     s=np.asarray(s)
    #     s.sort()
    #     at = s[0]
    #     dfOp.loc[dfOp[dfOp.Ativo == ativo].index, 'Ativo'] = at
    dfOp = dfOp[['notaNr', 'Ativo', 'Oper', 'DT', 'Qtde', 'VlrUnit']]
    dfOp['Qtde'] = dfOp['Qtde'].abs()
  return dfOp, dfNotas


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

path = r"app/"
parseFolder = "toParse/"

maybe_mkdir(path + parseFolder)
maybe_mkdir(path + parseFolder + f'parsed/')

st.title('Conversor de Notas de Corretagem')
st.write('Conversor de Notas de Corretagem para Excel')


files = st.file_uploader(
  label="Arraste notas de corretagem (em PDF, no padrão B3/Sinacor)", type=([".pdf"]),
  accept_multiple_files=True,
  help='Arraste quantas notas de corretagem - em PDF e no padrão B3/Sinacor - quiser para cá e clique em converter')
if len(files):
  left_column, right_column = st.beta_columns([1,1])
  Proceed = left_column.button('Converter')
  if Proceed:
    message = right_column.text('aguarde: processando...')

    for i, file in enumerate(files):
      with open(path + parseFolder + f'{file.name}', 'wb') as fd:
        fd.write(file.getvalue())
    dfOp, dfNotas = updateNotasB3(path=path, parseFolder=parseFolder)
    if len(dfOp) and len(dfNotas):
      message.text('Pronto. ')

      dfOp
      dfNotas
      # right_column.text('aaa')

      tmp_download_link_dfNotas = download_link(dfNotas.reset_index(), 'dfNotas.csv', 'dfNotas')
      st.markdown(tmp_download_link_dfNotas, unsafe_allow_html=True)
      tmp_download_link_dfOp = download_link(dfOp, 'dfOp.csv', 'dfOp')
      st.markdown(tmp_download_link_dfOp, unsafe_allow_html=True)
    else:
      message.write('Nada encontrado para converter. '
                   'Verifique se o pdf está no padrão B3')

    for file in files:
      os.replace(path + parseFolder + f'{file.name}', path + parseFolder + f'parsed/{file.name}')

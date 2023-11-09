import numpy as np
from datetime import  datetime
import pdfplumber
import pandas as pd
import os
import B3toXLS.cfg as cfg
from .cfg import NOTAS_MAP, SYMBOLS_MAP

class read_nota_b3(object):
    def __init__(self):
        self._notas = pd.read_pickle(cfg.FILE_NOTAS) if os.path.isfile(cfg.FILE_NOTAS) else pd.DataFrame()
        self._operacoes = pd.read_pickle(cfg.FILE_OPER) if os.path.isfile(cfg.FILE_OPER) else pd.DataFrame()
        self._cpf = None
        self._qant = pd.DataFrame([], columns=['Q', 'P']) \
            if not os.path.isfile(cfg.FILE_QANT) else pd.read_csv(cfg.FILE_QANT).set_index('symbol')
        self._pacum = pd.DataFrame([], columns=['noraml', 'daytrade']) \
            if not os.path.isfile(cfg.FILE_PACUM) else pd.read_csv(cfg.FILE_PACUM).set_index('cpf')

    def check_updates(self):
        files = os.listdir(cfg.PATH_NOTAS)
        files  = [f'{cfg.PATH_NOTAS}/{file}' for file in files if file[-3:] == 'pdf']
        if not len(files):
            print('Nada de novo para adicionar.')
            return
        idx = self._notas.index
        self.extract_tables(files)
        idx_new = self._notas.index.difference(idx)
        if not len(idx_new):
            print('Não houve mudanças.')
            return
        if self._notas['verified'].all():
            self.save()
            print(f'Notas adicionadas:\n{idx_new}')
            for file in files:
                file_dest = f'{cfg.PATH_PARSED}/{file.split(cfg.PATH_NOTAS)[1]}'
                os.system(f'mv "{file}" "{file_dest}"')
        else:
            print('Há notas com erros:')
            print(self._notas[~self._notas['verified']])

    def save(self):
        self._notas.to_pickle(cfg.FILE_NOTAS)
        self._operacoes.to_pickle(cfg.FILE_OPER)
        self._qant.to_csv(cfg.FILE_QANT)
        self._pacum.to_csv(cfg.FILE_PACUM)

    def extract_tables(self, files:list):
        def parse_str(col):
            map = NOTAS_MAP[col]
            if isinstance(map, dict):
                txt = map.get('txt')
                idx = map.get('idx')
                idx_sign = map.get('idx_sign')
                sign = map.get('sign',1)
            else:
                txt = map
                idx = -2
                idx_sign = -1
                sign=1
            pos = text.find(txt)
            if pos<0: return
            s = text[pos:].split('\n')[0].split(' ')
            try:
                return sign * pd.to_numeric(s[idx].replace('.', '').replace(',', '.'), errors='coerce') * (
                    -1 if idx_sign and s[idx_sign] == 'D'  else 1)
            except:
                return np.NaN
        operacoes = pd.DataFrame()
        notas = pd.DataFrame()
        notas.index.name='nota_id'
        for file in files:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    # Extrai o texto da página atual
                    text = page.extract_text()
                    # print(text)

                    # Para extrair tabelas, você pode fazer
                    tables = page.extract_tables({'intersection_y_tolerance': 21})

                    # parse notas
                    try:
                        topo = tables[0][0][0].split('\n')
                        topo0 = topo[2].split(' ')
                        corretora = topo[3].split(' ')[0].lower()
                        nota_id = f'{corretora}_{topo0[0]}'
                        if nota_id in notas.index and page.page_number<=notas.at[nota_id,'pagina']:
                            print(f'skipping file/page: {file}/{page} - nota ({nota_id}) already read.')
                            continue
                        cpf = topo[10].split(' ')[-1]
                        conta = topo[10].split(' ')[0]
                        data = pd.to_datetime(topo0[2],format='%d/%m/%Y').date()
                        notas.loc[nota_id, ['data','cpf','conta','pagina','file']]=data,cpf,conta,page.page_number, file
                        for col in NOTAS_MAP:
                            notas.at[nota_id, col] = parse_str(col)

                        oper = find_operacoes(tables)
                        oper = pd.DataFrame(oper,
                                            columns=['q', 'negociacao', 'c/v', 'tipo_mercado', 'prazo', 'titulo',
                                                     'obs', 'Q', 'P', 'valor', 'd/c'])
                        oper['nota_id'] = nota_id
                        operacoes = pd.concat([operacoes, oper], ignore_index=True)
                        print(f'parsed: {nota_id}/{page.page_number}')
                    except Exception as e:
                        print(f'Error in file: {file} - {page} - {e}')
        notas['opção'] = notas.apply(lambda row: row['opcoes_vendas']!=0 or row['opcoes_compras']!=0, axis=1)

        # ajusta campos numéricos
        cols_nr_oper = ['Q','P', 'valor']
        def f(row):
            v = pd.to_numeric(row[col].replace('.', '').replace(',', '.'),errors='coerce')
            return -v if row['c/v']=='V' and col in ['Q', 'valor'] else v
        for col in cols_nr_oper:
            operacoes[col] = operacoes.apply(f, axis=1)

        # adiciona col "dt" (daytrade)
        operacoes['dt'] = operacoes['obs'].apply(lambda x: True if "D" in x else False)

        #check values
        self.verify_parsed_values()
        self._notas = pd.concat([self._notas, notas]).sort_values(by=['data'])
        self._operacoes = pd.concat([self._operacoes, operacoes])
        self._operacoes['data'] = self._operacoes['nota_id'].apply(lambda x: self._notas.at[x, 'data'])
        self._operacoes.sort_values(by=['data','Q'], ascending=[True, False], inplace=True)
        self._operacoes.reset_index(inplace=True, drop=True)
        self.symbols_map()

    def symbols_map(self):
        operacoes = self._operacoes
        operacoes['titulo'] = operacoes['titulo'].apply(lambda x: SYMBOLS_MAP[x] if x in SYMBOLS_MAP else x)


    def verify_parsed_values(self, tol:float=.02):
        operacoes = self._operacoes
        for nota_id in self._notas.index:
            nota = self._notas.loc[nota_id].fillna(0)
            oper=operacoes[operacoes['nota_id']==nota_id]
            liq_operacoes = nota['liq_operacoes']
            check_1 = round(oper['valor'].sum(),2)==-liq_operacoes
            check_2 = nota['tot_cblc']==liq_operacoes+nota['tx_liq']+nota['tx_reg']
            v3 = round(
                nota['tot_cblc']+nota['corretagem']+nota['emolumentos']+nota['tx_ana']+
                nota['tx_termo_opcoes']+nota['ir_dt'],2
            )
            check_3 = round(v3-tol,2)<=nota['liquido']<=round(v3+tol,2)
            self._notas.loc[nota_id, 'verified'] = all([check_1, check_2, check_3])

    def calc_notas(self):
        def get_sum(cols):
            return sum([nota[c] for c in cols])
        for cpf in self.get_cpfs():
            self.cpf=cpf
            notas = self.notas
            operacoes = self.oper
            for nota_id in notas.index:
                nota = notas.loc[nota_id]
                oper = operacoes[operacoes['nota_id']==nota_id].copy()
                vendas = abs(oper.query(f'not dt and valor<0')['valor'].sum())
                vendas_dt = abs(oper.query(f'dt and valor<0')['valor'].sum())
                compras = abs(oper.query(f'not dt and valor>0')['valor'].sum())
                compras_dt = abs(oper.query(f'dt and valor>0')['valor'].sum())
                tx_bov_cblc = -get_sum(['tx_liq','tx_reg','tx_ana','tx_termo_opcoes','emolumentos'])
                tipo = 'opcao' if nota['opção'] else 'vista'
                tot_normal = vendas+compras
                tot_dt = vendas_dt+compras_dt
                if tot_dt:
                    tx_normal = sum([round(cfg.TAXAS[taxa][tipo]*tot_normal,2) for taxa in cfg.TAXAS])
                    tx_dt = tx_bov_cblc-tx_normal
                elif tot_normal:
                    tx_normal = tx_bov_cblc
                    tx_dt = 0
                tx_normal = -nota['corretagem']/(tot_dt+tot_normal) + tx_normal/tot_normal if tx_normal else 0
                tx_dt = -nota['corretagem']/(tot_dt+tot_normal) + tx_dt/tot_dt if tot_dt else 0
                oper['custo'] = oper.apply(
                    lambda r: abs(r['valor'] * (tx_dt if r['dt'] else tx_normal)), axis=1)
                assert round(oper['custo'].sum(),2) == round(tx_bov_cblc+abs(nota['corretagem']),2)
                self._operacoes.loc[oper.index, 'custo'] = oper['custo']
            operacoes = self.oper
            qant = self.qant.groupby(level=0)
            operacoes['Q_acum'] = 0
            for titulo in operacoes['titulo'].drop_duplicates():
                print(titulo)
                for if_dayt in [True, False]:
                    oper = self.get_oper_tit(titulo, if_dayt, operacoes)
                    Q_ant = qant.sum().at[titulo, 'Q'] if titulo in qant.indices else 0
                    P_ant = qant.last().at[titulo, 'P'] if titulo in qant.indices else 0
                    oper['Q_acum'] = oper['Q'].cumsum() + Q_ant
                    oper['v_oper'] = oper['valor']+oper['custo']
                    oper['if_add_q'] = oper.apply(lambda r: r['Q']*(r['Q_acum']-r['Q'])>=0, axis=1)
                    oper['p_medio'] = oper.apply(lambda r: r['v_oper']/r['Q'], axis=1)
                    oper['p_medio_prev'] = oper['p_medio'].shift(1).fillna(0)

                    # calc preço medio
                    p_medio_prev = P_ant
                    oper['pnl'] = 0
                    for i, if_add_q, Q, Q_acum, v_oper in zip(
                            oper.index, oper['if_add_q'], oper['Q'], oper['Q_acum'], oper['v_oper']):
                        if if_add_q:
                            p_medio = (v_oper+p_medio_prev*(Q_acum-Q))/Q_acum
                            pnl = 0
                        else:
                            p_medio = p_medio_prev
                            Q_pnl = min(-Q, Q_acum-Q)
                            pnl = Q_pnl*(v_oper/Q) - Q_pnl*p_medio
                        p_medio_prev=p_medio
                        oper.at[i, 'p_medio'] = p_medio
                        oper.at[i, 'pnl'] = pnl

                    self._operacoes.loc[oper.index, ['Q_acum','pnl']] = oper[['Q_acum','pnl']]

    def get_monthly_results(self, dt_start=None, dt_end=None):
        oper = self.oper
        notas = self.notas
        dt_start = oper['data'].min() if dt_start is None else pd.to_datetime(dt_start)
        dt_end = oper['data'].max() if dt_end is None else pd.to_datetime(dt_end)
        period = pd.period_range(dt_start, dt_end,freq='M')
        df = pd.DataFrame()
        for day,month,year in zip(period.day,period.month, period.year):
            dt_end_m = datetime(year, month, day).date()
            operm = oper[(oper['data'] >= datetime(year, month, 1).date())
                         & (oper['data'] <= dt_end_m)]
            df.loc[dt_end_m, ['normal','daytrade']] = operm.query('not dt')['pnl'].sum(), operm.query('dt')['pnl'].sum()
        return df

    def get_oper_tit(self, symbol, if_dayt:bool=None, operacoes:pd.DataFrame=None):
        '''

        Args:
            symbol (): B3 symbol
            if_dayt (): Day Trade
            operacoes (): Dataframe

        Returns: DF

        '''
        operacoes = self.oper if operacoes is None else operacoes
        return operacoes.query(f'titulo=="{symbol}" and {"dt" if if_dayt else "not dt"}').copy()

    @property
    def titulos(self):
        return self._operacoes['titulo'].drop_duplicates().to_list()

    def get_titulos_unmaped(self):
        return list(set([t for t in self._operacoes['titulo'] if t not in SYMBOLS_MAP.values()]))

    def filter_by_nota_id(self, nota_id):
        return self._notas.loc[nota_id], self._operacoes[self._operacoes['nota_id'] == nota_id]

    def filter_month(self, dt):
        dt = pd.to_datetime(dt)
        per = pd.period_range(dt,dt, freq='M')[0]
        dt_start = datetime(per.year, per.month, 1).date()
        dt_end = datetime(per.year, per.month, per.day).date()
        oper = self.oper
        return oper[(oper['data'] >= dt_start) & (oper['data'] <= dt_end)]


    def get_cpfs(self):
        return self._notas['cpf'].drop_duplicates().to_list()

    @property
    def cpf(self):
        return self._cpf

    @cpf.setter
    def cpf(self, value:str):
        self._cpf = value

    @property
    def notas(self):
        if self._cpf:
            return self._notas[self._notas['cpf']==self._cpf].sort_values(by=['data']).copy(True)
        else:
            print('cpf não informado.')

    @property
    def oper(self):
        notas = self.notas
        if notas is None: return
        return (self._operacoes[self._operacoes['nota_id'].isin(notas.index)].copy())

    @property
    def qant(self):
        if self._cpf:
            return self._qant[self._qant['cpf']==self.cpf]
        else:
            print('cpf não informado.')

    def carteira(self):
        oper = self.oper
        notas = self.notas
        qant = self.qant
        idx = list(set(oper['titulo'].drop_duplicates()).union(qant.index))
        contas = list(set(notas['conta'].drop_duplicates()).union(qant['conta'].drop_duplicates()))
        cart = pd.DataFrame(index=idx)
        for conta in contas:
            nota_ids = notas.query(f'conta=="{conta}"').index
            cart[conta] = 0
            cart[conta] = cart.apply(lambda x: oper[
                (oper['titulo']==x.name) & (oper['nota_id'].isin(nota_ids))]['Q'].sum(),axis=1)
            qant0 = qant[qant['conta']==conta]
            cart.loc[qant0.index, conta] += qant0['Q']
        return cart

    def get_nota(self, nota_id):
        return self._notas.loc[nota_id]

    def symbols_map_inverted_dict(self):
        return {v:[k for k in cfg.SYMBOLS_MAP if cfg.SYMBOLS_MAP[k]==v] for v in cfg.SYMBOLS_MAP.values()}

    def find_symbols(self, symbols):
        b3i = pd.read_csv(cfg.FILE_INSTRUMENTOS).set_index('TckrSymb')
        idx = b3i.index.intersection(symbols)
        b3i = b3i.loc[idx,['SpcfctnCd','CrpnNm']]
        to_map = self.get_titulos_unmaped()
        for tm in to_map:
            tms = tm.split(' ')
            tms0 = tms[0]
            fi = b3i[b3i['CrpnNm'].str.contains(tms0)==True].index
            while not len(fi) and len(tms0)>=3:
                tms0=tms0[:-1]
                fi = b3i[b3i['CrpnNm'].str.contains(tms0) == True].index
            fis = ''
            for i in fi:
                fis += f"'{i}' "
            print(f"'{tm}': {fis[:-1]},")



def find_indices_of_string(nested_list, substring, path=()):
    indices = []
    for index, element in enumerate(nested_list):
        current_path = path + (index,)
        if isinstance(element, list):
            indices.extend(find_indices_of_string(element, substring, current_path))
        elif isinstance(element, str) and substring in element:
            indices.append(current_path)
    return indices

def find_operacoes(tables):
    matching_lists = []
    for item in tables:
        # Check if the item is a list
        if isinstance(item, list):
            # Now check if the length is 11
            item = [i for i in item if not i is None]
            if len(item) == 11:
                # Ensure the third element is a string and check if it's 'C' or 'V'
                if isinstance(item[2], str) and item[2] in ['C', 'V']:
                    matching_lists.append(item)
            # If the item is another nested list, apply the function recursively
            else:
                matching_lists.extend(find_operacoes(item))
    return matching_lists


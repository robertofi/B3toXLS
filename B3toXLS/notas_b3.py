from B3toXLS.cfg import NOTAS_MAP,NOTAS_MAP_B3_V2
from utils import toList, create_unique_name
import warnings
import numpy as np
from datetime import  datetime, timedelta
import pdfplumber
import pandas as pd
import os
import B3toXLS.cfg as cfg
import json
from openpyxl import load_workbook

class notas_b3(object):
    def __init__(self):
        self._notas = pd.read_pickle(cfg.FILE_NOTAS) if os.path.isfile(cfg.FILE_NOTAS) else pd.DataFrame()
        self._operacoes = pd.read_pickle(cfg.FILE_OPER) if os.path.isfile(cfg.FILE_OPER) else pd.DataFrame(columns=['nota_id'])
        self._cpf = None
        self._qant = pd.DataFrame([], columns=['Q', 'P']) \
            if not os.path.isfile(cfg.FILE_QANT) else pd.read_csv(cfg.FILE_QANT).set_index('symbol')
        self._pacum = pd.DataFrame([], columns=['noraml', 'daytrade', 'date']) \
            if not os.path.isfile(cfg.FILE_PACUM) else pd.read_csv(cfg.FILE_PACUM).set_index('cpf')
        self._pacum['date'] = pd.to_datetime(self._pacum['date'])
        self.read_symbols_map()

    def read_symbols_map(self):
        global SYMBOLS_MAP
        with open(cfg.FILE_TO_SYMBOLS_MAP,'r') as file:
            SYMBOLS_MAP = json.load(file)

    def check_updates(self):
        files = os.listdir(cfg.PATH_NOTAS)
        files  = [f'{cfg.PATH_NOTAS}/{file}' for file in files if file[-3:] == 'pdf']
        if not len(files):
            print('Nada de novo para adicionar.')
            return
        idx = self._notas.index
        res = self.extract_tables(files)
        idx_new = self._notas.index.difference(idx)
        if not len(idx_new):
            print('Não houve mudanças.')
            return
        if self._notas['verified'].all():
            while len(self.get_unmaped_symbols()):
                print(f'Symbols unmaped:\n {[{k:"" for k in self.get_unmaped_symbols()}]}')
                print(f'Map the symbols at the file: {cfg.FILE_TO_SYMBOLS_MAP}')
                input('Press Enter to continue.')
                self.symbols_map()
            self.calc_notas(**res)
            self.save()
            print(f'Notas adicionadas:\n{idx_new}\n')
            for file in files:
                file_dest = f'{cfg.PATH_PARSED}/{file.split(cfg.PATH_NOTAS)[1]}'
                os.system(f'mv "{file}" "{file_dest}"')
        else:
            print('Há notas com erros:')
            print(self._notas[self._notas['verified']==False])

    def save(self):
        self._notas.to_pickle(cfg.FILE_NOTAS)
        self._operacoes.to_pickle(cfg.FILE_OPER)
        self._qant.to_csv(cfg.FILE_QANT)
        self._pacum.to_csv(cfg.FILE_PACUM)

    def criar_nota_de_ajuste(self, dt, conta):
        values = {'pagina':1,'file':'','corretagem':0,'liquido':0,'ir_oper':0,'emolumentos':0,'tx_oper':0,
         'tx_liq':0,'tx_reg':0,'tx_ana':0,'tx_termo_opcoes':0,'execucao':0,'ir_dt':0,'impostos':0,
         'outros':0,'liq_operacoes':0,'tot_cblc':0,'debentures':0,'opcoes_vendas':0,
         'opcoes_compras':0,'vendas_a_vista':0,'compras_a_vista':0,'opção':0,'verified':0}
        if self.cpf is None:
            raise Exception('Defina um CPF')
        nota_id = create_unique_name('aj',['aj'] + self._notas.index.to_list(), format='03d')
        self._notas.loc[nota_id, ['data', 'cpf', 'conta']] = pd.to_datetime(dt).date(), self.cpf, str(conta)
        self._notas.loc[nota_id, values.keys()] = values.values()
        return nota_id

    def delete_nota(self, nota_id):
        print(f'Notas with nota_id = {nota_id}:')
        print(self._notas.loc[[nota_id]])
        i = input(f'Confirm delete notas (Y/N)')
        if i.lower()=='y':
            self._notas.drop(nota_id, inplace=True)
            self._operacoes.drop(self._operacoes[self._operacoes['nota_id']==nota_id].index, inplace=True)

    def criar_oper_expirar_opcao(self, symbol, nota_id):
        nota = self.notas.loc[nota_id]
        oper = self.get_oper_tit(symbol)
        if not len(oper):
            warnings.warn('Título não encontrado')
            return
        oper=oper[oper['data']<=nota['data']]
        Q = -oper['Q'].sum()
        values = oper.iloc[-1].to_dict()
        values.update({'nota_id':nota_id,
                       'c/v': "V" if Q<0 else "C",
                       'Q':Q,
                       'P':0,
                       'valor':0,
                       'd/c': "C" if Q<0 else "D",
                       'dt':False,
                       'data':nota['data'],

                       'custo':0,
                       'Q_acum':0,
                       'pnl':0
                       })

        idx = self._operacoes.index.max() + 1
        self._operacoes.loc[idx] = pd.Series(values)

    def extract_tables(self, files:list):
        operacoes = pd.DataFrame(columns=['nota_id'])
        notas = pd.DataFrame()
        notas.index.name='nota_id'
        cpfs = []
        from_date = None
        for file in files:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    # Extrai o texto da página atual
                    for f in PARSE_NOTAS_FUNCS:
                        res = f(file,page)
                        if not res.get('success'):
                            continue
                    if not res.get('success'):
                        print(f'Error in file: {file} - {page} - {res.get("error")}')
                        continue
                    nota = res.get('nota')
                    oper = res.get('oper')
                    nota_id = nota.index[0]
                    data = nota.at[nota_id, 'data']
                    cpf = nota.at[nota_id, 'cpf']
                    if not cpf in cpfs:
                        cpfs.append(cpf)
                    if nota_id in notas.index and page.page_number <= notas.at[nota_id,'pagina']:
                        print(f'skipping file/page: {file}/{page} - nota ({nota_id}) already read.')
                        continue
                    from_date = data if from_date is None or data < from_date else from_date
                    operacoes = pd.concat([operacoes,oper],ignore_index=True)
                    notas.loc[nota_id, nota.columns] = nota.loc[nota_id]
                    print(f'parsed: {nota_id}/{page.page_number}')

        if not len(notas):
            print('Nenhuma nota adicionada')
            return
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
        notas_q = self._notas[~self._notas.index.isin(notas.index)]
        self._notas = pd.concat([notas_q, notas]).sort_values(by=['data'])
        self._notas['compras_a_vista'] = abs(self._notas['compras_a_vista'])
        self._notas['vendas_a_vista'] = abs(self._notas['vendas_a_vista'])
        operacoes_q = self._operacoes[~self._operacoes['nota_id'].isin(operacoes['nota_id'].drop_duplicates())]
        self._operacoes = pd.concat([operacoes_q, operacoes])
        self._operacoes['data'] = self._operacoes['nota_id'].apply(lambda x: self._notas.at[x, 'data'])
        self._operacoes.sort_values(by=['data','Q'], ascending=[True, False], inplace=True)
        self._operacoes.reset_index(inplace=True, drop=True)
        self.symbols_map()
        self.verify_parsed_values()
        return dict(cpfs=cpfs, from_date=from_date)

    def symbols_map(self):
        self.read_symbols_map()
        operacoes = self._operacoes
        operacoes['symbol'] = operacoes['titulo'].apply(lambda x: SYMBOLS_MAP[x] if x in SYMBOLS_MAP else x)
        unmapped = self.get_unmaped_symbols()
        if unmapped:
            print(f'Unmapped Symbols: \n {unmapped}')
            return False
        else:
            print('All Symbols Mapped')
            return True


    def verify_parsed_values(self, tol:float=.02):
        for nota_id in self._notas.index:
            check_1, check_2, check_3 = self.verify_nota_id(nota_id, tol)
            self._notas.loc[nota_id, 'verified'] = all([check_1, check_2, check_3])

    def verify_nota_id(self, nota_id, tol:float=.02, ):
        operacoes = self._operacoes
        nota = self._notas.loc[nota_id].fillna(0)
        oper = operacoes[operacoes['nota_id'] == nota_id]
        liq_operacoes = nota['liq_operacoes']
        check_1 = round(oper['valor'].sum(), 2) == -liq_operacoes
        tot_cblc_calc = round(liq_operacoes + nota['tx_liq'] + nota['tx_reg'], 2)
        check_2 = nota['tot_cblc'] == tot_cblc_calc
        if not check_2:
            print(f"{nota_id}: not passed in chcek 2. tot_cblc:{nota['tot_cblc']} "
                  f"!= {tot_cblc_calc} (tot_cblc_calc)")

        vs_cols = ['tot_cblc','corretagem','emolumentos','tx_ana','tx_termo_opcoes','ir_dt']

        v3 = round(nota[vs_cols].sum(), 2)
        check_3 = round(v3 - tol, 2) <= nota['liquido'] <= round(v3 + tol, 2)
        if not check_3:
            print(f'{nota_id}: not passed in chcek 3. v3:{v3} != {nota["liquido"]} (liquido para)')
            print(nota[vs_cols])
            print(f'tot_cblc+corretagem+emolumentos+tx_ana+tx_termo_opcoes !=  liquido')
        return check_1, check_2, check_3

    def set_expired_tit(self, titulo:str, year:int):
        oper = self.get_oper_tit(titulo)
        oper = oper[oper['data'].apply(lambda x: x.year == year)]
        if not len(oper): # chack if there is anything
            print('Não encontrado.')
            return
        for conta in oper['conta']:
            #get expiration date
            exp_date = oper['prazo'].iloc[0]
            assert (oper['prazo']==exp_date).all()
            exp_date = exp_date.split('/')
            exp_date = datetime(int(exp_date[1]),int(exp_date[0]),25 )
            if exp_date>datetime.now(): # check if date passed
                print(f'please wait till {exp_date} to do this operation at conta {conta}.')
                continue
            nota_idx = create_unique_name('ajuste', self._notas.index)
            self._notas.columns
            self.oper

    def calc_notas(self, cpfs:(str,list)=None, from_date:(str, datetime)=None):
        def get_sum(cols):
            return sum([nota[c] for c in cols])
        cpfs = toList(cpfs) if cpfs else [self.cpf] if self.cpf else self.get_cpfs()
        from_date = pd.to_datetime(from_date) if from_date else min(self._notas['data'])
        for cpf in cpfs:
            self.cpf=cpf
            notas = self.notas
            operacoes = self.oper
            for nota_id in notas.index:
                if notas.at[nota_id, 'data']<from_date:
                    continue
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
                tx_normal = -nota['corretagem']/(tot_dt+tot_normal) + tx_normal/tot_normal if tx_normal and (tot_dt+tot_normal) else 0
                tx_dt = -nota['corretagem']/(tot_dt+tot_normal) + tx_dt/tot_dt if tot_dt else 0
                oper['custo'] = oper.apply(
                    lambda r: abs(r['valor'] * (tx_dt if r['dt'] else tx_normal)), axis=1)
                assert round(oper['custo'].sum(),2) == round(tx_bov_cblc+abs(nota['corretagem']),2)
                self._operacoes.loc[oper.index, 'custo'] = oper['custo']
            operacoes = self.oper
            qant = self.qant.groupby(level=0)
            operacoes['Q_acum'] = 0
            for symbol in operacoes['symbol'].drop_duplicates():
                print(symbol)
                for if_dayt in [True, False]:
                    oper = self.get_oper_tit(symbol, if_dayt, operacoes)
                    Q_ant = qant.sum().at[symbol, 'Q'] if symbol in qant.indices else 0
                    P_ant = qant.last().at[symbol, 'P'] if symbol in qant.indices else 0
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
                    self._operacoes.loc[oper.index, ['Q_acum','pnl','p_medio']] = oper[['Q_acum','pnl','p_medio']]

    def get_monthly_results(self):
        #%%
        oper = self.oper
        notas = self.notas
        pacum = self._pacum.loc[self.cpf]
        dt_start = pacum['date'].date()
        dt_end = oper['data'].max()
        period = pd.period_range(dt_start, dt_end,freq='M')
        df = pd.DataFrame(columns=pd.MultiIndex.from_product([['normal','daytrade'],[]]))
        idx = pd.IndexSlice
        pacum_norm = df.at[dt_start, idx['normal','resultado']]=pacum['normal']
        pacum_dayt = df.at[dt_start, idx['daytrade','resultado']]=pacum['daytrade']
        ir_norm_acum = 0
        ir_dayt_acum = 0
        for day,month,year in zip(period.day,period.month, period.year):
            dt_end_m = datetime(year, month, day).date()
            if day==dt_start.day and month==dt_start.month and year==dt_start.year:
                df.loc[dt_end_m, idx['daytrade', 'prej_acum']] = pacum_dayt
                df.loc[dt_end_m, idx['normal', 'prej_acum']] = pacum_norm
                continue
            if month==1:
                ir_norm_acum = 0
                ir_dayt_acum = 0
            dt_start_m = datetime(year, month, 1).date()
            operm = oper[(oper['data'] >= dt_start_m) & (oper['data'] <= dt_end_m)]
            notasm = notas[(notas['data'] >= dt_start_m) & (notas['data'] <= dt_end_m)]
            df.loc[dt_end_m, idx['normal', 'ir_pago']] = notasm['ir_oper'].sum()
            ir_norm_acum += df.loc[dt_end_m, idx['normal', 'ir_pago']]
            df.loc[dt_end_m, idx['daytrade', 'ir_pago']] = notasm['ir_dt'].sum()
            ir_dayt_acum += df.loc[dt_end_m, idx['daytrade', 'ir_pago']]
            if notasm['vendas_a_vista'].abs().sum()< MIN_VALOR_ISENTO:
                res_isento = operm.query('dt==False')[(
                    operm['nota_id'].isin(notasm.query('opção==False').index))]['pnl'].sum()
                if res_isento:
                    df.loc[dt_end_m, idx['normal', 'isento']] = res_isento
            else:
                res_isento = 0
            res_norm = operm.query('dt==False')['pnl'].sum()-res_isento
            res_dayt = operm.query('dt')['pnl'].sum()
            df.loc[dt_end_m, idx['normal', 'resultado']] = res_norm
            df.loc[dt_end_m, idx['normal', 'res_trib']] = max(0,res_norm-res_isento+pacum_norm)
            df.loc[dt_end_m, idx['daytrade', 'resultado']] = res_dayt
            df.loc[dt_end_m, idx['daytrade', 'res_trib']] = max(0,res_dayt + pacum_dayt)
            ir_dev = (-df.loc[dt_end_m, idx['normal', 'res_trib']] * 0.15 -
                      df.loc[dt_end_m, idx['daytrade', 'res_trib']] * 0.2)
            if ir_dev:
                df.loc[dt_end_m, idx['total', 'ir_dev']] = ir_dev
                df.loc[dt_end_m, idx['total', 'pago']] = ir_dayt_acum + ir_norm_acum

                ir_dev -= ir_dayt_acum + ir_norm_acum
                ir_norm_acum = 0
                ir_dayt_acum = 0
            df.loc[dt_end_m, idx['total', 'DARF']] = ir_dev
            pacum_norm = min(res_norm+pacum_norm,0)
            pacum_dayt = min(res_dayt + pacum_dayt,0)
            df.loc[dt_end_m, idx['daytrade', 'prej_acum']] = pacum_dayt
            df.loc[dt_end_m, idx['normal', 'prej_acum']] = pacum_norm
        df.sort_index(axis=1, ascending=False, inplace=True)
        df.fillna(0, inplace=True)
        #%%
        return df

    def filter_by_date(self,dt_start:datetime, dt_end:datetime=None, if_one_month:bool=True):
        dt_start = pd.to_datetime(dt_start)
        oper = self.oper
        notas = self.notas
        if if_one_month:
            dt_start, dt_end = self._get_month_start_and_end_dates(dt_start)
        elif dt_end:
            dt_end=pd.to_datetime(dt_end)
        else:
            dt_end = oper['data'].max()
        dt_start = dt_start.date()
        dt_end = dt_end.date()
        operm = oper[(oper['data'] >= dt_start) & (oper['data'] <= dt_end)]
        notasm = notas[(notas['data'] >= dt_start) & (notas['data'] <= dt_end)]
        return notasm, operm

    def _get_month_start_and_end_dates(self, dt):
        dt = pd.to_datetime(dt)
        dt_start = datetime(dt.year, dt.month, 1)
        dt_end = datetime(dt.year + (0 if dt.month < 12 else 1),
                          ((dt.month + 1) if dt.month < 12 else 1), 1) - timedelta(
            days=1)
        return  dt_start,dt_end

    def get_oper_tit(self, symbol, if_dayt:bool=None, operacoes:pd.DataFrame=None):
        '''

        Args:
            symbol (): B3 symbol
            if_dayt (): Day Trade
            operacoes (): Dataframe

        Returns: DF

        '''
        operacoes = self.oper if operacoes is None else operacoes
        return operacoes.query(f'symbol=="{symbol}" and {"dt" if if_dayt else "not dt"}').copy()

    def get_pnl_tit(self, symbols, if_dayt:bool=None):
        return [rn.get_oper_tit(s, if_dayt)['pnl'].sum() for s in toList(symbols)]

    def get_darf(self, dt:(str, datetime)=None):
        if dt:
            dt = pd.to_datetime(dt)
        else:
            dt = datetime.now()
            dt=dt.replace(month= dt.month-1 if dt.month>1 else 12)
        df = self.get_monthly_results()
        dt_start, dt_end=self._get_month_start_and_end_dates(dt)
        valor = df[df.index==dt_end.date()][('total','DARF')].values[0]
        print(f'''
        https://sicalc.receita.economia.gov.br/sicalc/rapido/contribuinte
        cpf: {self.cpf}
        cod receita: 6015
        valor: {abs(round(valor,2))}
        periodo: {dt_end:%m/%Y}
        ''')

    @property
    def titulos(self):
        return self._operacoes['titulo'].drop_duplicates().to_list()

    @property
    def symbols(self):
        return self._operacoes['symbol'].drop_duplicates().to_list()

    def get_unmaped_symbols(self):
        return list(set([t for t in self._operacoes['symbol'] if t not in SYMBOLS_MAP.values()]))

    def filter_by_nota_id(self, nota_id):
        return self._notas.loc[nota_id], self._operacoes[self._operacoes['nota_id'] == nota_id]

    def show_last_notas_date(self):
        for cpf in self.get_cpfs():
            d=self._notas[self._notas['cpf']==cpf]['data'].max()
            print(cpf,d)
        df = self._notas.groupby(by='cpf')
        df[['data']].last()
        return df


    def filter_month(self, dt):
        dt = pd.to_datetime(dt)
        per = pd.period_range(dt,dt, freq='M')[0]
        dt_start = datetime(per.year, per.month, 1).date()
        dt_end = datetime(per.year, per.month, per.day).date()
        oper = self.oper.sort_values('data')
        return oper[(oper['data'] >= dt_start) & (oper['data'] <= dt_end)]

    def get_cpfs(self):
        return self._notas['cpf'].drop_duplicates().to_list()

    def set_cpf(self, cpf:(int, str)=None):
        if cpf is None:
            print(f'CPFs: \n {pd.DataFrame(self.get_cpfs())}')
            return
        self.cpf = self.get_cpfs()[cpf] if isinstance(cpf, int) else cpf
        print(f'CPF = {self.cpf}')

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

    def carteira(self, conta:str=None):
        oper = self.oper
        notas = self.notas
        qant = self.qant
        idx = list(set(oper['symbol'].drop_duplicates()).union(qant.index))
        contas = self.get_contas()
        if conta:contas=[c for c in contas if c == conta]
        cart = pd.DataFrame(index=idx)
        def f(x):
            try:
                return notas.at[oper.query(f'symbol=="{x.name}"')['nota_id'].values[0],'opção']
            except:
                return False
        for conta in contas:
            nota_ids = notas.query(f'conta=="{conta}"').index
            cart[conta] = 0
            cart[conta] = cart.apply(lambda x: oper[
                (oper['symbol']==x.name) & (oper['nota_id'].isin(nota_ids))]['Q'].sum(),axis=1)
            qant0 = qant[qant['conta']==conta]
            cart.loc[qant0.index, conta] += qant0['Q']
        cart['opção'] = cart.apply(lambda x: f(x),axis=1)

        # set p_medio for symbols in database
        idx = cart.index.intersection(oper['symbol'])
        cart.loc[idx, 'P']=[self.get_oper_tit(i)['p_medio'].values[-1] for i in idx]

        # set p_medio for symbols from previous operations
        idx = cart.index.difference(oper['symbol'])
        cart.loc[idx, 'P']=qant.loc[idx, 'P']

        cart = cart[(cart[contas]!=0).any(axis=1)]
        return cart.sort_index()

    def get_contas(self):
        return list(set(self.notas['conta'].drop_duplicates()).union(self.qant['conta'].drop_duplicates()))

    def save_carteira_to_rtd(self):
        file = f'{cfg.PATH_TO_RTD_XLS}/carteira_{self.cpf}.xlsx'
        cart = self.carteira()
        contas = self.get_contas()
        cart['Q'] = cart[ contas].sum(axis=1)
        cart.drop(contas, inplace=True, axis=1)
        cart_acoes = cart[~cart['opção']].drop('opção', axis=1)
        cart_opcoes = cart[cart['opção']].drop('opção', axis=1)

        try:
            book = load_workbook(file)
        except FileNotFoundError:
            book = None

        with pd.ExcelWriter(file,engine='openpyxl') as writer:
            if book:
                writer.book = book
            # Write your DataFrames as new sheets
            cart_acoes.to_excel(writer,sheet_name='ações',index=True)
            cart_opcoes.to_excel(writer,sheet_name='opções',index=True)
        # df.to_excel(file)
        print(f'file saved at: {file}')

    def compare_carteira_with_real(self, path_to_posicao):
        '''
        compara posicao real com carteira no bco de dados
        Args:
            path_to_posicao (): arquivo em excel, com culnas: symbol e conta (iguais a carteira)

        Returns: posicoes com diferenca (se vier vazio, não há diferença)

        '''
        posicao = pd.read_excel(path_to_posicao, sheet_name='posicao').set_index('symbol').drop_duplicates()

        posicao.columns = posicao.columns.astype(str)
        dif = posicao.copy()
        cart = self.carteira()
        for col in dif.columns:
            if col in cart.columns:
                i = dif.index.intersection(cart.index)
                dif[col] = dif.loc[i,[col]].apply(lambda x: x[col] - cart.at[x.name,col], axis =1)
                for i in cart.index.difference(posicao.index):
                    dif.loc[i, col]=-cart.loc[i, col]
        dif=dif[(dif!=0).any(axis=1)]
        i = dif.index.intersection(posicao.index)
        cols = dif.columns.copy()
        for col in cols:
            c=f'real_{col}'
            dif[c]=0
            dif.loc[i, c] = posicao.loc[i, col]
        i = dif.index.intersection(cart.index)
        for col in cols:
            c = f'cart_{col}'
            dif[c]=0
            dif.loc[i, c] = cart.loc[i, col]
        si = self.symbols_map_inverted_dict()
        for i in dif.index:
            dif.loc[i, 'map']=str(si.get(i))
        return dif

    def get_nota(self, nota_id):
        return self._notas.loc[nota_id]

    def symbols_map_inverted_dict(self):
        return {v:[k for k in B3toXLS.symbols_map.SYMBOLS_MAP if B3toXLS.symbols_map.SYMBOLS_MAP[k] == v] for v in B3toXLS.symbols_map.SYMBOLS_MAP.values()}

    def find_symbols(self, symbols):
        b3i = pd.read_csv(cfg.FILE_INSTRUMENTOS).set_index('TckrSymb')
        idx = b3i.index.intersection(symbols)
        b3i = b3i.loc[idx,['SpcfctnCd','CrpnNm']]
        to_map = self.get_unmaped_symbols()
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


MIN_VALOR_ISENTO = 20000


def find_indices_of_string(nested_list, substring, path=()):
    indices = []
    for index, element in enumerate(nested_list):
        current_path = path + (index,)
        if isinstance(element, list):
            indices.extend(find_indices_of_string(element, substring, current_path))
        elif isinstance(element, str) and substring in element:
            indices.append(current_path)
    return indices


def find_operacoes(tables, len_items=11,):
    pos_cant_be_empty = [1,2,3,5,7,8,9,10]
    matching_lists = []
    for item in tables:
        # Check if the item is a list
        if isinstance(item, list):
            # Now check if the length is len_items
            item = [i for i in item if not i is None]
            if '1-BOVESPA' in item:
                # Ensure the third element is a string and check if it's 'C' or 'V'
                while len(item)>11:
                    empty_pos = [i for i,j in enumerate(item) if not j]

                    rem = set(pos_cant_be_empty).intersection(empty_pos)
                    if len(rem):
                        item.pop(list(rem)[0])
                    else:
                        item.pop(empty_pos[-1])

                if isinstance(item[2], str) and item[2] in ['C', 'V']:
                    matching_lists.append(item)
            # If the item is another nested list, apply the function recursively
            else:
                matching_lists.extend(find_operacoes(item))
    return matching_lists


def parse_str(text, col, MAP):
    map = MAP[col]
    if isinstance(map,dict):
        txt = map.get('txt')
        idx = map.get('idx', -2)
        idx_sign = map.get('idx_sign',-1)
        sign = map.get('sign',None)
        split_idx = map.get('split_idx',0)
        data_type = map.get('type', 'number')
    else:
        txt = map
        idx = -2
        idx_sign = -1
        sign = None
        split_idx = 0
        data_type = 'number'
    pos = text.find(txt)
    if pos < 0: return
    s = text[pos:].split('\n')[split_idx].split(' ')
    try:
        sign=sign if sign else -1 if idx_sign and s[idx_sign] == 'D' else 1
        return (sign * pd.to_numeric(s[idx].replace('.','').replace(',','.'))
                if data_type=='number'
                else str(s[idx]) if data_type=='str'
                else pd.to_datetime(s[idx]) if data_type=='datetime'
                else s[idx])
    except:
        if 'second_try' in map:
            map = map['second_try']
            idx = map.get('idx')
            idx_sign = map.get('idx_sign')
            sign = map.get('sign',1)
            try:
                return (sign * pd.to_numeric(s[idx].replace('.','').replace(',','.')) * (
                    -1 if idx_sign and s[idx_sign] == 'D' else 1) if data_type=='number'
                else str(s[idx]) if data_type=='str'
                else pd.to_datetime(s[idx]) if data_type=='datetime'
                else s[idx])
            except:
                pass
        return (np.NaN if data_type=='number'
                else '' if data_type=='str'
                else pd.NaT if data_type=='datetime'
                else None)


def parse_nota_b3_v1(file,page):
        text = page.extract_text()
        # print(text)
        tables = page.extract_tables({'intersection_y_tolerance':21})
        nota = pd.DataFrame()
        nota.index.name='nota_id'
        # parse nota
        try:
            topo = tables[0][0][0].split('\n')
            topo0 = topo[2].split(' ')
            corretora = topo[3].split(' ')[0].lower()
            nota_id = f'{corretora}_{topo0[0]}'
            cpf = topo[10].split(' ')[-1]
            conta = topo[10].split(' ')[0]
            data = pd.to_datetime(topo0[2],format='%d/%m/%Y').date()
            nota.loc[nota_id,['data','cpf','conta','pagina',
                               'file']] = data,cpf,conta,page.page_number,file
            for col in NOTAS_MAP:
                nota.at[nota_id,col] = parse_str(text, col, NOTAS_MAP)
            oper = find_operacoes(tables)
            oper = pd.DataFrame(oper,
                                columns=['q','negociacao','c/v','tipo_mercado','prazo','titulo',
                                         'obs','Q','P','valor','d/c'])
            oper['nota_id'] = nota_id
            return dict(success=True, oper=oper, nota=nota)
        except Exception as e:
            return dict(success=False, error=e)


def parse_nota_b3_v2(file,page):
    text = page.extract_text()
    # print(text)
    nota = pd.DataFrame()
    nota.index.name='nota_id'
    # parse nota
    try:
        corretora = parse_str(text,'corretora',NOTAS_MAP_B3_V2)
        nota_id = parse_str(text,'nota_id',NOTAS_MAP_B3_V2)
        data = parse_str(text,'data',NOTAS_MAP_B3_V2)
        conta = parse_str(text,'conta',NOTAS_MAP_B3_V2)
        cpf = parse_str(text,'cpf',NOTAS_MAP_B3_V2)
        nota_id = f'{corretora.lower()}_{nota_id}'
        data = pd.to_datetime(data,format='%d/%m/%Y').date()
        nota.loc[
            nota_id,['data','cpf','conta','pagina','file']] = data,cpf,conta,page.page_number,file
        for col in NOTAS_MAP:
            nota.at[nota_id,col] = parse_str(text, col, NOTAS_MAP)

        # find total value
        pos0 = text.find('Quantidade Preço / Ajuste Valor Operação / Ajuste D/C')
        pos1 = text.find('Resumo dos Negócios Resumo Financeiro')
        oper_t = text[pos0:pos1].split('\n')[1:-1]
        valor_total_ver = sum(
            [pd.to_numeric(t.split(' ')[-2].replace('.','').replace(',','.')) for t in oper_t])

        oper = page.within_bbox((0,243,595,446)).extract_table(
            {'intersection_tolerance':20,"vertical_strategy":"lines","horizontal_strategy":"text",
                "snap_y_tolerance":5,'text_x_tolerance':30})
        cols = ['q','negociacao','c/v','tipo_mercado','prazo','titulo','obs',
                                     'Q','P','valor','d/c']
        for i,row in enumerate(oper):
            if len(row)<len(cols) and row[0]=='1-BOVESPA':
                oper[i] = ['']+row

        oper = pd.DataFrame(oper,columns=cols)
        valor_total = pd.to_numeric(oper.valor.str.replace('.','').str.replace(',','.')).sum()

        oper['nota_id'] = nota_id
        if round(valor_total_ver,2)!=round(valor_total,2):
            print('Error in sum')

        return dict(success=True, oper=oper, nota=nota)
    except Exception as e:
        return dict(success=False, error=e)


PARSE_NOTAS_FUNCS = [parse_nota_b3_v1, parse_nota_b3_v2]

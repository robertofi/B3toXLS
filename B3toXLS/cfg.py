
import pathlib
PATH = str(pathlib.Path(__file__).parent.resolve())+'/'
PATH_ROOT = str(pathlib.Path(PATH).parent)+'/'
PATH_DATA = '/etc/tekton_data/b3'
PATH_NOTAS = f'{PATH_DATA}/notas'
PATH_PARSED = f'{PATH_NOTAS}/parsed'
PATH_TO_RTD_XLS = '/home/r/Documents'
FILE_NOTAS = f'{PATH_DATA}/notas.pkl'
FILE_OPER = f'{PATH_DATA}/operacoes.pkl'
FILE_EVENTOS = f'{PATH_DATA}/eventos.pkl'
FILE_INSTRUMENTOS = f'{PATH_DATA}/b3instruments.csv'
FILE_QANT = f'{PATH_DATA}/posicao_anterior.csv'
FILE_PACUM = f'{PATH_DATA}/prejuizo_acum.csv'
FILE_TO_SYMBOLS_MAP = f'{PATH}/symbols_map.json'


NOTAS_MAP = dict(
    corretagem = 'Total Custos / Despesas',
    liquido = 'Líquido para',
    ir_oper = dict(txt='I.R.R.F. s/ operações', idx = -2,idx_sign = -1,sign=1,
                   second_try=dict(idx = -1, sign=-1)),
    emolumentos = 'Emolumentos',
    tx_oper = 'Taxa Operacional',
    tx_liq = 'Taxa de liquidação',
    tx_reg = dict(txt='Taxa de Registro',sign=-1,second_try=dict(idx = -1,sign=-1)),
    tx_ana = dict(txt='Taxa A.N.A.',second_try=dict(idx = -1)),
    tx_termo_opcoes = dict(txt='Taxa de termo/opções',second_try=dict(idx = -1)),
    execucao = dict(txt = 'Execução', idx = 1) ,
    ir_dt = dict(txt = 'IRRF Day Trade:', idx = -4, sign=-1) ,
    impostos = 'Impostos' ,
    outros = 'Outros',
    liq_operacoes = 'Valor líquido das operações',
    tot_cblc = 'Total CBLC',
    debentures = dict(txt = 'Debêntures', idx = 1),
    opcoes_vendas = dict(txt = 'Opções - vendas', idx = 3),
    opcoes_compras = dict(txt = 'Opções - compras', idx = 3),
    vendas_a_vista = dict(txt = 'Vendas à vista', idx = 3),
    compras_a_vista = dict(txt = 'Compras à vista', idx = 3),
)
NOTAS_MAP_B3_V2 = dict(
    data=dict(txt='Nr. nota Folha Data pregão', idx = 2, split_idx=1,type='str'),
    cpf=dict(txt='Cliente C.P.F./C.N.P.J/C.V.M./C.O.B.', idx = -1, split_idx=1,type='str'),
    conta=dict(txt='Cliente C.P.F./C.N.P.J/C.V.M./C.O.B.', idx = 0, split_idx=1,type='str'),
    nota_id=dict(txt='Nr. nota Folha Data pregão', idx = 0, split_idx=1,type='str'),
    corretora=dict(txt='Nr. nota Folha Data pregão', idx = 0, split_idx=2,type='str')
)



TAXAS = dict(
    negociacao = dict(vista = 0.00005, opcao = 0.00037, vista_dt = 0.00005, opcao_dt = 0.00037),
    liquidacao = dict(vista = 0.00025, opcao = 0.000275, vista_dt = 0.00018, opcao_dt = 0.00018),
    registro = dict(vista = 0., opcao = 0.000695, vista_dt = 0., opcao_dt = 0.00015),
)


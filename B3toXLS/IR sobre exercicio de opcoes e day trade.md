# IR sobre exercicio de opcoes e day trade

Atualizado em: 2026-09-11

Este documento registra a interpretacao usada pelo projeto para classificar, para fins de IR, operacoes de exercicio de opcoes que aparecem na nota de corretagem com observacao `D`.

## Resumo pratico

Operacoes do tipo `EXERC OPC COMPRA` e `EXERC OPC VENDA` nao devem ser casadas automaticamente como day trade apenas porque a nota trouxe `D` na coluna de observacao ou porque houve compra/venda do ativo objeto no mesmo pregao.

Para fins fiscais, o exercicio de opcoes tem regra propria. No projeto, essas linhas devem ser tratadas como operacoes comuns de exercicio de opcoes, e nao como day trade automatico de acao.

## Por que a nota mostra `D`

Na legenda da nota de corretagem, `D` indica day trade. Em algumas notas, a corretora marca com `D` operacoes ligadas ao exercicio quando ha compra e venda do mesmo ativo no mesmo pregao. Isso faz sentido do ponto de vista operacional da nota e da liquidacao.

Mas essa marcacao nao deve ser usada como criterio fiscal absoluto quando o `tipo_mercado` e `EXERC OPC COMPRA` ou `EXERC OPC VENDA`. Nesses casos, a Receita trata o resultado em regras especificas de exercicio de opcoes.

## Fundamento fiscal

A Receita Federal define day trade, de forma geral, como operacao ou conjunto de operacoes iniciadas e encerradas no mesmo dia, com o mesmo ativo, na mesma instituicao intermediadora, com quantidade liquidada total ou parcialmente.

Fonte: Receita Federal, "Bolsa de Valores":
https://www.gov.br/receitafederal/pt-br/assuntos/meu-imposto-de-renda/pagamento/renda-variavel/bolsa-de-valores-1

No entanto, o material oficial de Perguntas e Respostas IRPF 2026 trata separadamente o exercicio de opcoes:

Fonte: Receita Federal, Perguntas e Respostas IRPF 2026:
https://www.gov.br/receitafederal/pt-br/centrais-de-conteudo/publicacoes/perguntas-e-respostas/dirpf/p-r-irpf-2026-v1-00-2026-04-23.pdf

Itens relevantes:

- Questao 725: ganho liquido no exercicio de opcoes de compra.
- Questao 726: ganho liquido no exercicio de opcoes de venda.

Na questao 725, para titular de opcao de compra, o custo de aquisicao e o preco de exercicio do ativo acrescido do premio pago. Se houver venda a vista do ativo na data do exercicio, o ganho liquido e a diferenca positiva entre o valor da venda a vista e esse custo de aquisicao.

Para o lancador de opcao de compra, o ganho liquido e a diferenca positiva entre o preco de exercicio, acrescido do premio recebido, e o custo de aquisicao do ativo entregue. Se o lancador estava descoberto, o custo de aquisicao e o preco pago para adquirir o ativo objeto do exercicio.

Na questao 726, a Receita tambem da exemplo de exercicio de opcao de venda com ordem simultanea de compra no mercado a vista, e ainda assim trata o caso dentro da regra de exercicio de opcoes.

## Exemplo: compra de BOVA11 para honrar exercicio

Caso observado:

- Nota: `xp_133788908`
- Data: `2026-04-02`
- Ativo objeto: `BOVA11`
- Operacao de exercicio: `V EXERC OPC COMPRA`
- Compra a vista no mesmo dia: `C VISTA ISHARES BOVA CI`
- A nota veio com `obs = D`/`D#`

Antes da correcao, o sistema casava:

- compra de 2.000 BOVA11 no mercado a vista;
- venda/entrega de 2.000 BOVA11 por exercicio de opcao.

Como as quantidades batiam no mesmo dia, tudo era marcado como `dt=True`.

Essa classificacao e inadequada para fins fiscais porque a venda decorre de exercicio de opcao. O resultado deve seguir a regra de exercicio de opcoes, nao a regra generica de day trade no mercado a vista.

## Tratamento correto no projeto

Regra implementada:

- Linhas cujo `tipo_mercado` comeca com `EXERC OPC` nao entram no casamento automatico de day trade.
- Essas linhas sao forcadas para `dt=False`.
- A compra ou venda a vista associada ao exercicio tambem nao deve ser casada contra a linha de exercicio.
- A compra/venda a vista so pode virar day trade se houver outra operacao comum, independente do exercicio, que a case no mesmo dia.

Em termos de codigo, a normalizacao de day trade deve calcular o casamento intradiario somente sobre operacoes "matchable", excluindo exercicios de opcoes.

## Impacto na apuracao

Se o exercicio de opcao gerar prejuizo e nao for day trade:

- o prejuizo fica em operacoes comuns;
- pode compensar ganhos futuros de operacoes comuns em renda variavel, conforme a segregacao aplicavel;
- nao compensa ganhos de day trade;
- nao deve ser usado para reduzir imposto de FII ou outras categorias separadas.

Se o mes ficar negativo em operacoes comuns, nao ha DARF de operacoes comuns naquele mes. O prejuizo acumulado pode reduzir a base tributavel de meses seguintes, dentro da mesma categoria de apuracao.

No caso revisado de abril/2026, apos excluir `EXERC OPC` do casamento automatico de day trade, a apuracao passou a carregar prejuizo em operacoes comuns de opcoes, reduzindo a base tributavel do mes seguinte.

## Observacao sobre o termo "swing trade"

Para o codigo e para relatorios fiscais, o termo preferido e `operacao comum`, nao `swing trade`.

No uso de mercado, "swing trade" costuma significar compra e venda em dias diferentes. Ja para IR, a separacao principal no demonstrativo e entre:

- operacoes comuns;
- day trade.

Exercicio de opcao que nao se enquadra como day trade deve ficar no grupo de operacoes comuns, com a regra propria de opcoes.

## Regra de seguranca para futuras manutencoes

Nao usar `obs` contendo `D` como criterio unico e absoluto de day trade.

Antes de marcar `dt=True`, verificar o `tipo_mercado`:

- `EXERC OPC COMPRA`: excluir de day trade automatico.
- `EXERC OPC VENDA`: excluir de day trade automatico.
- `OPCAO DE COMPRA` e `OPCAO DE VENDA`: podem ser day trade se houver compra e venda da mesma serie no mesmo pregao.
- `VISTA`: pode ser day trade se houver compra e venda comum do mesmo ativo no mesmo pregao, desconsiderando contrapartes oriundas de exercicio de opcoes.

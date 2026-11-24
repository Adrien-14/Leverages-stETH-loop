# Leverages-stETH-loop
Deposit stETH, borrow ETH, buy stETH, repeat. The question is what it costs to
defend the position during a peg wobble once you account for market depth:
most loop calculators sell collateral at the oracle price, here the rebalance
goes through a stETH/ETH x*y=k pool so slippage scales with size vs depth.

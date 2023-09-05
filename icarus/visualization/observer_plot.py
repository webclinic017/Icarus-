import finplot as fplt

def quote_asset2(dashboard_data, ax):
    fplt.plot(dashboard_data['quote_asset']['total'], width=3, ax=ax, legend='Total')
    fplt.plot(dashboard_data['quote_asset']['free'], width=2, ax=ax, legend='Free')
    fplt.plot(dashboard_data['quote_asset']['in_trade'], width=2, ax=ax, legend='In Trade')
    fplt.add_line((dashboard_data['quote_asset']['total'].index[0], dashboard_data['quote_asset']['total'].iloc[0]),
        (dashboard_data['quote_asset']['total'].index[-1], dashboard_data['quote_asset']['total'].iloc[0]), color='#000000', interactive=False)


def quote_asset(x, y, axes):
    axes['ax'].reset()
    axes['axo'].reset()
    axes['ax_bot'].reset()
    axes['axo_bot'].reset()
    
    disable_ax_bot(axes)
    fplt.plot(y['total'], width=3, ax=axes['ax'], legend='Total')
    fplt.plot(y['free'], width=2, ax=axes['ax'], legend='Free')
    fplt.plot(y['in_trade'], width=2, ax=axes['ax'], legend='In Trade')
    fplt.add_line((y['total'].index[0], y['total'].iloc[0]),
        (y['total'].index[-1], y['total'].iloc[0]), color='#000000', interactive=False)


def disable_ax_bot(axes):
    axes['ax'].set_visible(xaxis=True)
    axes['ax_bot'].hide()

def quote_asset_leak(dashboard_data, ax):
    fplt.plot(dashboard_data['quote_asset_leak']['binary'], width=3, ax=ax, legend='binary')


def enable_ax_bot(axes, **kwargs):
    fplt._ax_reset(axes['ax_bot'])

    axes['ax'].set_visible(xaxis=False)
    axes['ax_bot'].show()

    #if kwargs.get('reset', True): fplt._ax_reset(axes['ax_bot'])
    if y_range := kwargs.get('y_range', None): fplt.set_y_range(y_range[0], y_range[1], ax=axes['ax_bot'])
    if band := kwargs.get('band', None): fplt.add_band(band[0], band[1], color='#6335', ax=axes['ax_bot'])



def text(x, y, axes):
    fplt.plot(x, y=[1]*len(x), ax=axes['ax_bot'])
    for index, row in y.iterrows():
        fplt.add_text((index, 0.5), str(row[0]), color='#000000',anchor=(0,0), ax=axes['ax_bot'])
    pass
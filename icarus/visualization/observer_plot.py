import finplot as fplt

def quote_asset(dashboard_data, ax):
    fplt.plot(dashboard_data['quote_asset']['total'], width=3, ax=ax, legend='Total')
    fplt.plot(dashboard_data['quote_asset']['free'], width=2, ax=ax, legend='Free')
    fplt.plot(dashboard_data['quote_asset']['in_trade'], width=2, ax=ax, legend='In Trade')
    fplt.add_line((dashboard_data['quote_asset']['total'].index[0], dashboard_data['quote_asset']['total'].iloc[0]),
        (dashboard_data['quote_asset']['total'].index[-1], dashboard_data['quote_asset']['total'].iloc[0]), color='#000000', interactive=False)


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
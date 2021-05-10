import webbrowser

b = webbrowser.get()

html_index = 'http://www.github.com/MilanSkocic'
print(html_index)
fpath = html_index

b.open(fpath)
SCRIPTS = sc18_figure10.py sc18_figure2.py sc18_figure3_figure4.py sc18_figure5.py sc18_figure6_figure7.py sc18_figure8_figure9.py

all: $(SCRIPTS)

%.py: %.ipynb
	jupyter nbconvert --to script "$*.ipynb" --stdout > "$@"
	sed -i "/get_ipython/d" "$@" 

clean:
	rm -rvf $(SCRIPTS)

SCRIPTS = sc18-figure10.py sc18-figure2.py sc18-figure3-figure4.py sc18-figure5.py sc18-figure6-figure7.py sc18-figure8-figure9.py

all: $(SCRIPTS)

%.py: %.ipynb
	jupyter nbconvert --to script "$*.ipynb" --stdout > "$@"
	sed -i "/get_ipython/d" "$@" 

clean:
	rm -rvf $(SCRIPTS)

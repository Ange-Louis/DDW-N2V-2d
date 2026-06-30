import typer

from src.ddw.fit_model2 import fit_model2
from src.ddw.prepare_data2 import prepare_data
from src.ddw.refine_tomogram2 import refine_tomogram

from src.ddw.fit_n2v_model import fit_n2v_model
from src.ddw.prepare_n2v_data import prepare_n2v_data
from src.ddw.refine_n2v_tomogram import refine_n2v_tomogram

# pretty_exceptions_show_locals=False gives shorter error messages
app = typer.Typer(pretty_exceptions_show_locals=False)
app.command()(prepare_data)
app.command()(fit_model2)
app.command()(refine_tomogram)

app.command()(prepare_n2v_data)
app.command()(fit_n2v_model)
app.command()(refine_n2v_tomogram)


def main():
    app()


if __name__ == "__main__":
    main()

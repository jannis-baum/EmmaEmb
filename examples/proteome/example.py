from ema.embedding.get_embeddings import get_embeddings
from ema.utils import read_fasta

file_path_sequences = "examples/proteome/uniprot_sprot_human.fasta"


def main():

    sequences = read_fasta(file_path=file_path_sequences)
    proteins = list(sequences.keys())[:100]
    get_embeddings(
        input=proteins, model="esm2_t48_15B_UR50D", output_dir="embeddings"
    )

    print()


if __name__ == "__main__":
    main()

from emma.utils import read_fasta, write_fasta

path_swissprot = "examples/proteome/uniprot_sprot.fasta"
path_swissprot_filtered = "examples/proteome/uniprot_sprot_human.fasta"


def main():
    """
    Filter SwissProt database for human proteins.
    Save the filtered database to a new file.
    """

    sequences = read_fasta(path_swissprot)
    sequences = {
        protein.split("|")[1]: sequence
        for protein, sequence in sequences.items()
        if "Homo sapiens" in protein
    }
    write_fasta(file_path=path_swissprot_filtered, sequences=sequences)


if __name__ == "__main__":
    main()

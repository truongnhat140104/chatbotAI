from app.core.loader import HotichLoader
from app.core.indexer import HotichIndexer

loader = HotichLoader()
bundle = loader.load_all()

indexer = HotichIndexer(bundle)
result = indexer.build_all()
result.print_summary()
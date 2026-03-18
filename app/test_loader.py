from app.core.loader import HotichLoader

loader = HotichLoader()
bundle = loader.load_all()
bundle.print_summary()
bundle.print_messages(level="ERROR")
bundle.print_messages(level="WARNING", limit=20)
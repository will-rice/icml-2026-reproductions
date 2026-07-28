import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: quick_validate.py <skill_dir>")
        sys.exit(1)
    skill_dir = Path(sys.argv[1])
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        print(f"Error: {skill_md} does not exist")
        sys.exit(1)
    print(f"Validated skill at {skill_dir}")

if __name__ == "__main__":
    main()
